#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""l0_swimlane — generate an AICore intra-core swimlane trace.json for a task.

Given a SceneTest test file + platform + a comma-list of func_ids (the mix
member set), this tool:
  1. runs (or reuses) a JSON-only args dump (`--dump-args 3`) to capture the
     task's real per-arg metadata,
  2. picks the task whose active-subtask set == `--func-id` and reconstructs its
     FULL positional args[] (shapes / dtypes / strides / start_offset / scalar
     values) from the #1181 dump; the task's func_id ARRAY is its mix membership,
  3. generates the "intra-core replay" workspace — a single combined
     `replay_entry` whose cube sub-core runs the AIC member and vec sub-core(s)
     the AIV member, so a MIX task replays AS ONE OP (msprof op simulator recipe
     from .claude/skills/insight-trace/SKILL.md),
  4. smoke-builds it, then runs `msprof op simulator` and exports the task's
     trace.json (a combined AIC+AIV swimlane for a mix).

Mix support: any mix (same- or different-source members) replays as one task.
`--func-id` IS the member set — name the task's FULL set: `--func-id 0` traces
a task the orchestration dispatches on its own (a single-core task {0}),
`--func-id 0,1,2` traces a 3-way mix in full. The author wrote the
orchestration, so the member set is known directly — there is no shape-guessing.
LIMITATION
(tier C): cross-core synchronisation that the orchestration drove (task deps /
barriers) is absent in this isolated replay, so inter-core waits are optimistic
— the per-core pipeline structure is faithful, the AIC<->AIV handoff timing is
not. 2-AIV mixes (e.g. func_id 0,1,2) route per AIV lane via the hardware
get_subblockid(); validated on the a2a3sim camodel (the two AIV lanes run
different kernels, e.g. VADD vs VMUL).

Usage (onboard — recommended; wrap the whole tool in one task-submit so the dump
and the camodel collect share the locked device):
    task-submit --device auto --run \
        "python -m simpler_setup.tools.l0_swimlane \
            --test tests/st/a2a3/.../test_paged_attention_unroll.py \
            --func-id 0 --platform a2a3 --case <small case>"

Both the dump (when `--platform` is onboard) and the msprof collect step need an
NPU device context. Run the tool under a single outer task-submit: it appends
--device <id> to this command (also exported as $TASK_DEVICE); that one device
threads through both the dump and the collect, so neither grabs a second lock.
With a sim `--platform` (a2a3sim/a5sim) the dump needs no NPU, and the collect
self-locks via its own task-submit. Requires ASCEND_HOME_PATH (source CANN
set_env.sh first).
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.platform_info import parse_platform
from simpler_setup.pto_isa import ensure_pto_isa_root

# Dump emits dtype as an UPPERCASE string (src/common/task_interface/data_type.h
# get_dtype_name). Map string -> (DataType raw enum value, element bytes).
DTYPE_RAW = {
    "FLOAT32": 0,
    "FLOAT16": 1,
    "INT32": 2,
    "INT16": 3,
    "INT8": 4,
    "UINT8": 5,
    "BFLOAT16": 6,
    "INT64": 7,
    "UINT64": 8,
    "UINT16": 9,
    "UINT32": 10,
}
DTYPE_SIZE = {
    "FLOAT32": 4,
    "FLOAT16": 2,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT8": 1,
    "BFLOAT16": 2,
    "INT64": 8,
    "UINT64": 8,
    "UINT16": 2,
    "UINT32": 4,
}

# --set-arg tensor fill writes an integer into every element, so it only makes
# sense for integer-typed control tensors (loop counts / indices like
# context_lens). Filling a float/bf16 tensor with an int is almost always a
# mistake, so it is refused.
INTEGER_DTYPES = frozenset(
    {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }
)

KARGS_SLOTS = 50  # MAX_TENSOR_ARGS(16) + MAX_SCALAR_ARGS(32) + 2

# Per-arch build parameters. soc = msprof/simulator SoC version (and the
# $CANN/aarch64-linux/simulator/<soc>/lib path); aicore_arch = bisheng
# --cce-aicore-arch (single mix-arch, no -cube/-vec suffix); cce / npu_arch are
# the standalone-compile prologue macros the kernel headers expect.
# aiv_lanes_per_block = AIV lanes per AICore cluster (the hardware subblockdim).
# Both a2a3 and a5 are 1C2V (1 AIC + 2 AIV per cluster), so this is 2; it sizes
# the AIV context rows in the mix replay. It is a hardware constant, not a
# per-kernel property — hence it lives here, not in the test.
ARCH_CONFIG = {
    "a2a3": {
        "soc": "dav_2201",
        "aicore_arch": "dav-c220",
        "cce": 220,
        "npu_arch": "PTO_NPU_ARCH_A2A3",
        "aiv_lanes_per_block": 2,
    },
    "a5": {
        "soc": "dav_3510",
        "aicore_arch": "dav-c310",
        "cce": 310,
        "npu_arch": "PTO_NPU_ARCH_A5",
        "aiv_lanes_per_block": 2,
    },
}

# --- SPMD context + mix detection (no per-test markers needed) --------------
# SPMD kernels read an execution context at args slots 48/49 —
# LocalContext{block_idx, block_num} (48) and GlobalContext{sub_block_id} (49) —
# which the orchestration builds per dispatch. l0_swimlane runs an isolated
# replay (no orchestration), so it SYNTHESIZES that context host-side. None of
# its inputs need a per-test marker; they are all derived:
#   * is-mix      — a cooperative mix kernel is the SAME source compiled for both
#                   sub-cores, so it appears as an (aic, aiv) incore PAIR sharing
#                   one source. Detected by load_kernel_meta; everything else
#                   (incl. independent kernels packed into a mix dispatch, like
#                   mixed_example) goes through the AIC/AIV-only path.
#   * hw_block_dim / block_num — the case's `block_dim` (the SPMD grid width).
#   * aiv_lanes_per_block      — the arch's hardware subblockdim (ARCH_CONFIG).
# The mix path additionally needs ONE incore to declare the full tensor
# `signature` (so the dump captures the shared args) — a standard CALLABLE
# field, not a tool-specific one.


# ---------------------------------------------------------------------------
# Step 1: read kernel metadata (source path + core_type) from the test file
# ---------------------------------------------------------------------------
def _first_platform_case(cls, platform):
    """The case l0 auto-pins when `--case` is omitted: the name of the FIRST
    `CASES[*]` that lists `platform` in its `platforms` (manual or not — l0
    always dumps with `--manual include`). None if no case lists the platform.
    Auto-pinning one case makes the dump deterministic (no "run all cases,
    reconstruct from the newest dump dir" ambiguity), and ties the synthesized
    slot-48 block_num to the SAME case the dump ran (block_dim resolved via the
    caller's per-case map)."""
    for c in getattr(cls, "CASES", []):
        if platform in c.get("platforms", []):
            return c.get("name")
    return None


def load_kernel_meta(test_path: Path, func_id: int, platform: str):
    spec = importlib.util.spec_from_file_location("l0_swimlane_testmod", str(test_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import test module from {test_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @scene_test's inspect.getfile(cls) can resolve the
    # class -> module -> file path for relative CALLABLE source resolution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from simpler_setup import SceneTestCase  # noqa: PLC0415

    classes = [
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CALLABLE")
    ]
    if not classes:
        raise ValueError(f"No SceneTestCase with CALLABLE found in {test_path}")

    # Build a func_id -> metadata lookup across every incore of every class. With
    # the #1181 dump the mix membership is read from the dump's func_id array (a
    # task property), not guessed here from a shared source — so this is a flat
    # per-func lookup, and the codegen resolves each mix member through it.
    #
    # CALLABLE `source` is relative to the test file's directory (e.g.
    # "../../mixed_example/kernels/aic/kernel_matmul.cpp"). Resolve each to an
    # ABSOLUTE path so the generated replay_kernel.cpp can `#include` it from the
    # workspace dir, and so a mix whose members live in different dirs each
    # resolve correctly.
    by_func = {}
    owner_cls = {}
    for cls in classes:
        for inc in cls.CALLABLE.get("incores", []):
            fid = inc["func_id"]
            by_func[fid] = {
                "func_id": fid,
                "source": (test_path.parent / inc["source"]).resolve(),
                "core_type": inc["core_type"],
                "name": inc.get("name") or Path(inc["source"]).stem,
            }
            owner_cls[fid] = cls
    if func_id not in by_func:
        avail = ", ".join(f"{f}={m['name']}({m['core_type']})" for f, m in sorted(by_func.items()))
        raise ValueError(f"func_id={func_id} not found. Available incores: {avail}")
    tgt = by_func[func_id]
    cls = owner_cls[func_id]
    auto_case = _first_platform_case(cls, platform)
    # name -> block_dim for every case, so main can resolve block_dim from the
    # case actually selected (--case X, or the auto-pinned first-platform case),
    # not an arbitrary CASES entry. A case declaring no block_dim maps to 1.
    block_dim_by_case = {
        c.get("name"): int(c.get("config", {}).get("block_dim") or 1) for c in getattr(cls, "CASES", [])
    }
    return {
        "by_func": by_func,
        "target_func_id": func_id,
        "source": tgt["source"],
        "core_type": tgt["core_type"],
        "name": tgt["name"],
        "class_name": cls.__name__,
        "auto_case": auto_case,
        "block_dim_by_case": block_dim_by_case,
    }


def _case_from_manifest(manifest: Path, class_name: str) -> str:
    """Recover the case name from the dump dir `<ClassName>_<Case>_<YYYYMMDD>_<HHMMSS>`."""
    run_dir = manifest.parent.parent.name  # args_dump's parent = the run dir
    m = re.match(rf"{re.escape(class_name)}_(.+)_\d{{8}}_\d{{6}}$", run_dir)
    return m.group(1) if m else "case"


# ---------------------------------------------------------------------------
# Step 2: obtain an args_dump.json (run the test in sim, or reuse one)
# ---------------------------------------------------------------------------
def get_or_run_dump(test_path: Path, platform: str, variant: str, dump_json, case=None, device=None):
    if dump_json:
        p = Path(dump_json)
        if not p.is_file():
            raise FileNotFoundError(f"--dump-json not found: {p}")
        return p

    outputs = PROJECT_ROOT / "outputs"
    before = set(outputs.glob("*/args_dump")) if outputs.is_dir() else set()
    # Level 3 (full, JSON-only): every task's tensor *metadata* (shape/dtype/
    # strides) + scalar values, no .bin payload copy. That is exactly what arg
    # reconstruction consumes (it never reads the payload), and it skips the
    # device->host arena copy entirely — cheaper, and avoids the large-shape
    # copy failing onboard.
    cmd = [sys.executable, str(test_path), "-p", platform, "--dump-args", "3"]
    # Pin the dump to exactly one case, allowing it to be `manual` (l0_swimlane
    # tracing targets are often manual to stay out of CI). `case` is main's
    # resolved case: the explicit --case, else the auto-pinned first-platform
    # case. Pinning one case keeps the dump deterministic — the reconstruction
    # can't pick the wrong (newest) dump dir when several cases ran. `case` is
    # None only when the test declares no case on this platform (single-case /
    # no-CASES); then the dump runs the test as-is.
    if case:
        cmd += ["--case", case, "--manual", "include"]
    # Onboard dump runs on a real NPU and must use the task-submit-locked device.
    # When the whole tool is wrapped in an outer task-submit, that device reaches
    # us via the appended --device <id> (resolved into `device` by main) — thread
    # it into the test so the dump and the later camodel collect share the one
    # lock instead of grabbing separate devices. Sim variants need no device.
    if variant != "sim":
        if device:
            cmd += ["--device", device]
        else:
            print(
                "[l0_swimlane] WARNING: onboard dump with no locked device — not "
                "under task-submit; the dump will use the framework default "
                "device unlocked. Wrap the whole tool in task-submit (see "
                ".claude/rules/running-onboard.md)."
            )
    print(f"[l0_swimlane] running dump: {' '.join(cmd[1:])}")
    # check=True on purpose: a golden PASS on the dump run is the free
    # validation that func_id / signature / args are wired correctly, so the
    # captured dump is trustworthy. A golden FAIL means the capture is suspect
    # — never reconstruct a trace from it.
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    after = set(outputs.glob("*/args_dump"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    # MUST come from THIS run. Never fall back to a pre-existing args_dump —
    # that would silently reconstruct args from an unrelated test's dump and
    # produce a wrong-but-passing trace. An empty `new` means the dump was
    # skipped (e.g. signature/payload mismatch -> "args dump skipped" /
    # "No args dump data to export" warnings above).
    if not new:
        raise RuntimeError(
            f"dump for {test_path.name} produced no NEW outputs/*/args_dump — the "
            f"args dump was skipped (see the 'args dump skipped' / 'No args "
            f"dump data to export' warnings above). The incore signature likely does "
            f"not match the dispatched payload. Refusing to reuse a stale dump."
        )
    cand = new[-1]
    manifest = cand / "args_dump.json"
    if not manifest.is_file():
        raise RuntimeError(f"manifest missing (dump produced no data): {manifest}")
    return manifest


# ---------------------------------------------------------------------------
# Step 3: reconstruct one task's args from the #1181 dump (func_id ARRAY model)
# ---------------------------------------------------------------------------
def reconstruct_task_args(manifest: Path, func_id_list, task_id=None):
    """Reconstruct one task's full positional args[] from the #1181 dump.

    The dump (`args_dump.json`, top-level "args") stamps every record with the
    task's active-subtask membership as a func_id ARRAY (slot order AIC, AIV0,
    AIV1) and emits each payload slot once, positionally. The caller names the
    exact member set via `--func-id` (e.g. `0,1,2`); we pick the task whose
    func_id SET matches it and take its FULL payload (every slot, sorted by
    arg_index); each member kernel reads its own slice. Returns
    (chosen_task_id, tensor_count, args, mix_func_ids), with mix_func_ids the
    chosen task's func_id array AS STORED (slot order — NOT the typed order, so
    lane assignment stays correct).

    `task_id` pins a specific instance; default = lowest.
    """
    data = json.loads(manifest.read_text())
    entries = data["args"]

    def fids(t):
        return tuple(t.get("func_id") or [])

    want = set(func_id_list)
    recs = [t for t in entries if set(fids(t)) == want]
    if not recs:
        shapes = [list(s) for s in sorted({fids(t) for t in entries})]
        raise ValueError(f"--func-id {sorted(want)} matches no task; dump has func_id shapes {shapes}")

    tasks = sorted({t["task_id"] for t in recs})
    chosen = task_id if task_id is not None else tasks[0]
    if chosen not in set(tasks):
        raise ValueError(f"task_id {chosen} not among the selected tasks ({tasks})")
    trecs = [t for t in recs if t["task_id"] == chosen]
    mix_func_ids = list(fids(trecs[0]))  # slot order: AIC, AIV0, AIV1

    # Union both stages, keyed by arg_index (INOUT appears twice -> keep one).
    by_arg = {}
    for t in trecs:
        ai = t["arg_index"]
        if ai not in by_arg or t["stage"] == "before_dispatch":
            by_arg[ai] = t

    tensors = sorted((t for t in by_arg.values() if t["kind"] != "scalar"), key=lambda t: t["arg_index"])
    scalars = sorted((t for t in by_arg.values() if t["kind"] == "scalar"), key=lambda t: t["arg_index"])
    tensor_count = len(tensors)
    # Args need NOT start at arg_index 0 or be contiguous: a kernel dispatched as
    # a non-first MIX subtask reads its tensors at an offset (e.g. mixed_example's
    # ADD reads args[3..5], MUL reads args[6..8]). So validate only that every arg
    # slot is distinct and each tensor has a shape; the kernel reads args[slot],
    # and the replay places each tensor at its real slot (decoupled from the
    # 0-based descriptor-array index — see emit_replay_host).
    seen = set()
    for t in tensors:
        if not t.get("shape"):
            raise ValueError(f"tensor arg {t['arg_index']} has empty shape")
        if t["arg_index"] in seen:
            raise ValueError(f"duplicate tensor arg_index {t['arg_index']}")
        seen.add(t["arg_index"])
    for s in scalars:
        if s["arg_index"] in seen:
            raise ValueError(f"scalar arg_index {s['arg_index']} collides with a tensor slot")
        seen.add(s["arg_index"])

    args = []
    for t in tensors:
        dt = t["dtype"].upper()
        shape = list(t["shape"])
        strides = list(t.get("strides") or _row_major(shape))
        args.append(
            {
                "kind": "tensor",
                "slot": t["arg_index"],
                "dtype": dt,
                "shape": shape,
                "strides": strides,
                "start_offset": int(t.get("start_offset", 0)),
            }
        )
    for s in scalars:
        args.append({"kind": "scalar", "slot": s["arg_index"], "value": int(s["value"])})
    return chosen, tensor_count, args, mix_func_ids


def _row_major(shape):
    st = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        st[i] = acc
        acc *= shape[i]
    return st


def _extent_elem(shape, strides):
    e = 1
    for s, stride in zip(shape, strides):
        if s > 0:
            e += (s - 1) * stride
    return e


def _is_contiguous(shape, strides, start_offset):
    exp = 1
    for s, stride in zip(reversed(shape), reversed(strides)):
        if stride != exp:
            return False
        exp *= s
    return start_offset == 0


# ---------------------------------------------------------------------------
# Step 4: code-generation for the 5 workspace files
# ---------------------------------------------------------------------------
def _prologue(cfg) -> str:
    return f"""\
#ifndef __CCE_AICORE__
#define __CCE_AICORE__ {cfg["cce"]}
#endif
#include <cce_aicore_intrinsics.h>
#ifndef {cfg["npu_arch"]}
#define {cfg["npu_arch"]}
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 ((event_t)7)
#endif
#ifndef PIPE_FIX
#define PIPE_FIX ((pipe_t)10)
#endif
"""


def _split_cores(members):
    """Partition mix members into the AIC and AIV sub-cores (slot order preserved)."""
    aic = [m for m in members if m["core_type"] == "aic"]
    aiv = [m for m in members if m["core_type"] == "aiv"]
    return aic, aiv


def _member_entry_symbol(m):
    """Per-member renamed kernel entry symbol (B-tier separate compilation)."""
    return f"l0_f{m['func_id']}_entry"


def emit_replay_kernel_combined(members, cfg) -> str:
    """Combined replay_entry for a whole mix task (mix-together).

    `members` is the mix's incore metadata in func_id (slot) order — each
    {"func_id","core_type","source"(abs Path),"name"}.

    Two regimes by AIV count:

    * **A-tier (<=1 AIV)** — single translation unit: the AIC member is
      `#include`d under `#if defined(__DAV_CUBE__)`, the AIV member under
      `#if defined(__DAV_VEC__)`. Different arch variants of one mix-arch
      compile, so the two `kernel_entry`/`get_num_tiles` definitions never
      clash. The simulator runs both sub-cores from one `<<<1>>>` launch (one
      block = 1 AIC + 2 AIV), giving a combined swimlane. With one AIV member
      the vec body has no per-lane guard, so BOTH AIV lanes run it — redundant
      for a 1-AIV mix, but the faithful behavior for an SPMD mix whose two AIV
      lanes legitimately run the same kernel (a duplicate func_id collapsed
      above), each lane differing only by `get_subblockid()`.

    * **B-tier (2 AIV)** — still ONE translation unit, but the two AIV `.cpp`
      share file-scope names (`static get_num_tiles`, `extern "C" kernel_entry`),
      so each `#include` is wrapped in `#define`-renames + `#undef`
      (`kernel_entry`->`l0_f<id>_entry`, `get_num_tiles`->`l0_f<id>_get_num_tiles`).
      Both AIV includes sit under `#if defined(__DAV_VEC__)` so the vector ISA
      target feature is in scope (an un-guarded include compiles for the wrong
      arch — that is why separate compilation failed). `replay_entry` routes per
      sub-core, and per AIV lane via `get_subblockid()`.

    NOTE (tier C, PLAN §3.4): cross-core sync the orchestration drove is absent
    in this isolated replay, so inter-core waits are optimistic.
    """
    aic, aiv = _split_cores(members)
    # Two AIV members that share ONE source are the same program on both lanes —
    # whether the orchestration gave them the same func_id (spmd_paged_attention_highperf:
    # aiv0 = aiv1 = PA_AIV → dump `[0,1,1]`) or two distinct func_ids that compile
    # the SAME `.cpp` (spmd_multiblock_mix: func 1 & 2 both `kernel_spmd_mix.cpp` →
    # dump `[0,1,2]`). Collapse them by source: the A-tier single-AIV path
    # `#include`s the source ONCE and both lanes run it (the kernel self-routes by
    # sub-block id). Including the same source twice would redefine its file-scope
    # statics (the rename below only covers `kernel_entry` / `get_num_tiles`); only
    # DISTINCT-source AIV members need the 2-AIV rename path below.
    seen_aiv = set()
    aiv = [m for m in aiv if not (m["source"] in seen_aiv or seen_aiv.add(m["source"]))]
    if len(aic) > 1:
        raise NotImplementedError(f"{len(aic)} AIC members in one task; 1C2V has a single AIC sub-core")
    if len(aiv) > 2:
        raise NotImplementedError(f"{len(aiv)} distinct AIV members in one task; 1C2V has at most 2 AIV sub-cores")

    if len(aiv) <= 1:
        # A-tier: single-TU #include path (cube=AIC variant, vec=AIV variant).
        def arch_section(arch_macro, member):
            if member is None:
                return "", f"#if defined({arch_macro})\n    // no member on this sub-core.\n#endif"
            inc = f'#if defined({arch_macro})\n{_prologue(cfg)}#include "{member["source"]}"\n#endif\n'
            body = (
                f"#if defined({arch_macro})\n"
                f"    kernel_entry(args);  // func_id={member['func_id']} {member['name']} ({member['core_type']})\n"
                f"#endif"
            )
            return inc, body

        cube_inc, cube_body = arch_section("__DAV_CUBE__", aic[0] if aic else None)
        vec_inc, vec_body = arch_section("__DAV_VEC__", aiv[0] if aiv else None)
        desc = ", ".join(f"{m['name']}(func {m['func_id']},{m['core_type']})" for m in members)
        return f"""\
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Combined mix replay_entry — cube sub-core runs the AIC member, vec the AIV
// member; one mix-arch binary, the simulator runs each sub-core's path.
// Mix members (slot order): {desc}
{cube_inc}{vec_inc}
extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args) {{
{cube_body}
{vec_body}
}}
"""

    # B-tier (2 AIV): ONE TU like A-tier. The two AIV .cpp share file-scope names
    # (static get_num_tiles + extern "C" kernel_entry), so each #include is
    # wrapped in #define-renames + #undef. Both AIV includes go under
    # #if defined(__DAV_VEC__) so the vector ISA target feature is in scope (an
    # un-guarded include compiles for the wrong arch — that broke separate
    # compilation). The renamed entries are routed per sub-core / AIV lane.
    aiv0, aiv1 = aiv[0], aiv[1]

    def rename_block(m):
        # Rename every file-scope name the co-resident sources SHARE. Repo
        # convention: kernel_entry + get_num_tiles (the *_impl helpers already
        # differ, e.g. add_impl/mul_impl; standalone AIVs have no get_num_tiles,
        # so renaming it is a harmless no-op there). A future 2-AIV pair that
        # shares OTHER file-scope statics must add them to this list.
        return (
            f"#define kernel_entry {_member_entry_symbol(m)}\n"
            f"#define get_num_tiles l0_f{m['func_id']}_get_num_tiles\n"
            f'#include "{m["source"]}"\n'
            f"#undef get_num_tiles\n"
            f"#undef kernel_entry\n"
        )

    cube_inc = f"#if defined(__DAV_CUBE__)\n{_prologue(cfg)}{rename_block(aic[0])}#endif\n" if aic else ""
    vec_inc = f"#if defined(__DAV_VEC__)\n{_prologue(cfg)}{rename_block(aiv0)}{rename_block(aiv1)}#endif\n"
    if aic:
        cube_body = (
            f"#if defined(__DAV_CUBE__)\n"
            f"    {_member_entry_symbol(aic[0])}(args);  // AIC: func_id={aic[0]['func_id']} {aic[0]['name']}\n"
            f"#endif"
        )
    else:
        cube_body = "#if defined(__DAV_CUBE__)\n    // no AIC member in this mix.\n#endif"
    desc = ", ".join(f"{m['name']}(func {m['func_id']},{m['core_type']})" for m in members)
    return f"""\
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Combined mix replay_entry (B-tier: 2 AIV) — single TU. Each member #include is
// wrapped in #define-renames (kernel_entry -> l0_f<id>_entry, get_num_tiles ->
// l0_f<id>_get_num_tiles) + #undef so the two AIV kernels' shared file-scope
// names don't clash; each include sits under its sub-core arch guard so the ISA
// target feature is in scope. Mix members (slot order): {desc}
{cube_inc}{vec_inc}
extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args) {{
{cube_body}
#if defined(__DAV_VEC__)
    // Per-AIV-lane routing. RISK (issue #900 / runtime intrinsic.h): simpler's
    // runtime considers the CCE get_subblockid() unreliable (returns 0 for BOTH
    // AIV lanes because the runtime does not program that register). Whether a
    // bare camodel replay op returns the true physical lane (0=AIV0, 1=AIV1) is
    // UNVERIFIED — validate the trace shows veccore0 and veccore1 running
    // DIFFERENT kernels. get_sub_block_id(args) is NOT usable here: the replay's
    // single shared args[] gives both lanes the same slot-49 GlobalContext.
    if (get_subblockid() == 0) {{
        {_member_entry_symbol(aiv0)}(args);  // AIV0: func_id={aiv0["func_id"]} {aiv0["name"]}
    }} else {{
        {_member_entry_symbol(aiv1)}(args);  // AIV1: func_id={aiv1["func_id"]} {aiv1["name"]}
    }}
#endif
}}
"""


def emit_replay_launch() -> str:
    return """\
#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args);

// HW_BLOCK_NUM = 1: single task in isolation.
extern "C" void launch_replay(void *args, void *stream) {
    replay_entry<<<1, nullptr, stream>>>((__gm__ int64_t *)args);
}
"""


def _emit_tensor_alloc_descs(args):
    """Per-tensor (alloc, desc, argrow, free) C snippets.

    `ti` is the tensor's 0-based index in the descriptor array `d_tensors`;
    `slot` is its real args[] position. These differ when the kernel reads its
    tensors at an offset (a non-first MIX subtask, e.g. args[3..5]) — so the
    descriptor index and the args slot are decoupled.
    """
    alloc, descs, argrow, frees = [], [], [], []
    tensor_args = [a for a in args if a["kind"] == "tensor"]
    for ti, a in enumerate(tensor_args):
        slot = a["slot"]
        shape, strides, dt = a["shape"], a["strides"], a["dtype"]
        esz = DTYPE_SIZE[dt]
        buf_bytes = (a["start_offset"] + _extent_elem(shape, strides)) * esz
        contig = 1 if _is_contiguous(shape, strides, a["start_offset"]) else 0
        ndims = len(shape)
        shp = ", ".join(str(x) for x in shape)
        strd = ", ".join(str(x) for x in strides)
        # Default: data memset to 0 (only descriptor metadata is real). When
        # --set-arg fills this tensor, write VALUE into every element instead —
        # for control tensors whose CONTENT drives the kernel (e.g. paged
        # attention reads n_blocks from the context_lens tensor). The low `esz`
        # bytes of the int64 VALUE are copied per element (correct for any
        # integer width, little-endian).
        fill = a.get("fill")
        if fill is None:
            init = f"    ACL_CHECK(aclrtMemset(d_t{ti}, t{ti}Bytes, 0, t{ti}Bytes));"
        else:
            init = (
                f"    {{\n"
                f"        std::vector<unsigned char> hbuf{ti}(t{ti}Bytes, 0);\n"
                f"        const int64_t fillv{ti} = {fill}LL;\n"
                f"        for (size_t off = 0; off + {esz} <= t{ti}Bytes; off += {esz})\n"
                f"            memcpy(hbuf{ti}.data() + off, &fillv{ti}, {esz});\n"
                f"        ACL_CHECK(aclrtMemcpy(d_t{ti}, t{ti}Bytes, hbuf{ti}.data(), t{ti}Bytes,\n"
                f"                              ACL_MEMCPY_HOST_TO_DEVICE));\n"
                f"    }}"
            )
        alloc.append(
            f"    void *d_t{ti} = nullptr;\n"
            f"    const size_t t{ti}Bytes = {buf_bytes}ULL;\n"
            f"    ACL_CHECK(aclrtMalloc(&d_t{ti}, t{ti}Bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n"
            f"{init}"
        )
        descs.append(
            f"    {{\n"
            f"        const uint32_t shp[] = {{{shp}}};\n"
            f"        const uint32_t strd[] = {{{strd}}};\n"
            f"        make_desc(h_tensors.data() + {ti} * 128, (uint64_t)(uintptr_t)d_t{ti},\n"
            f"                  t{ti}Bytes, {a['start_offset']}ULL, shp, strd, {ndims}, {DTYPE_RAW[dt]}, {contig});\n"
            f"    }}"
        )
        argrow.append(f"    h_args[{slot}] = (int64_t)((uintptr_t)d_tensors + (size_t){ti} * 128);")
        frees.append(f"    aclrtFree(d_t{ti});")
    return alloc, descs, argrow, frees


def emit_replay_host(tensor_count: int, args, block_num: int = 1) -> str:
    alloc, descs, argrow, free_list = _emit_tensor_alloc_descs(args)
    for a in args:
        if a["kind"] == "scalar":
            argrow.append(f"    h_args[{a['slot']}] = (int64_t){a['value']}LL;  // scalar")
    frees = "\n".join(free_list)

    return f"""\
// Auto-generated by simpler_setup.tools.l0_swimlane — do not edit by hand.
// Builds the kernel's real args (from args dump) and launches replay_entry.
#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(expr)                                                        \\
    do {{                                                                       \\
        aclError _e = (expr);                                                  \\
        if (_e != ACL_SUCCESS) {{                                              \\
            fprintf(stderr, "ACL error %d at %s:%d\\n", _e, __FILE__, __LINE__);\\
            return 1;                                                          \\
        }}                                                                     \\
    }} while (0)

// 128B Tensor descriptor — offsets pinned by static_assert in tensor.h:
//   buffer.addr@0 buffer.size@8 start_offset@24 ndims@36 dtype@40
//   is_contiguous@42 shapes@44 strides@72.
static void make_desc(void *dst128, uint64_t dev_addr, uint64_t buf_bytes,
                      uint64_t start_offset, const uint32_t *shapes,
                      const uint32_t *strides, uint32_t ndims, uint8_t dtype_raw,
                      uint8_t is_contig) {{
    uint8_t b[128];
    memset(b, 0, sizeof(b));
    *reinterpret_cast<uint64_t *>(b + 0) = dev_addr;
    *reinterpret_cast<uint64_t *>(b + 8) = buf_bytes;
    *reinterpret_cast<uint64_t *>(b + 24) = start_offset;
    *reinterpret_cast<int32_t *>(b + 32) = 0;
    *reinterpret_cast<uint32_t *>(b + 36) = ndims;
    b[40] = dtype_raw;
    b[42] = is_contig;
    for (uint32_t i = 0; i < ndims; ++i)
        *reinterpret_cast<uint32_t *>(b + 44 + 4 * i) = shapes[i];
    for (uint32_t i = 0; i < ndims; ++i)
        *reinterpret_cast<uint32_t *>(b + 72 + 4 * i) = strides[i];
    memcpy(dst128, b, sizeof(b));
}}

extern "C" void launch_replay(void *args, void *stream);

int main() {{
    const char *dev_s = getenv("ACL_DEVICE_ID");
    int32_t device_id = dev_s ? atoi(dev_s) : 0;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    constexpr int kArgsSlots = {KARGS_SLOTS};
    constexpr int kNumTensors = {tensor_count};

{chr(10).join(alloc)}

    std::vector<uint8_t> h_tensors((size_t)kNumTensors * 128, 0);
{chr(10).join(descs)}

    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(&d_tensors, (size_t)kNumTensors * 128, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_tensors, (size_t)kNumTensors * 128, h_tensors.data(),
                          (size_t)kNumTensors * 128, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> h_args(kArgsSlots, 0);
{chr(10).join(argrow)}

    // SPMD context at slots 48/49 — built unconditionally. Harmless for
    // positional kernels (they ignore 48/49); required for SPMD kernels that
    // read get_block_idx / get_block_num / get_sub_block_id, which would
    // otherwise dereference a null context. block_idx=0 traces a representative
    // block; block_num={block_num} (the case's block_dim) keeps steady-state
    // branches (e.g. `block_idx+1 < block_num`) on their normal path.
    uint8_t h_local[64] = {{0}};   // LocalContext: block_idx@0, block_num@4
    *reinterpret_cast<int32_t *>(h_local + 0) = 0;
    *reinterpret_cast<int32_t *>(h_local + 4) = {block_num};
    uint8_t h_global[16] = {{0}};  // GlobalContext: sub_block_id@0 = 0
    void *d_local = nullptr, *d_global = nullptr;
    ACL_CHECK(aclrtMalloc(&d_local, sizeof(h_local), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_local, sizeof(h_local), h_local, sizeof(h_local), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMalloc(&d_global, sizeof(h_global), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_global, sizeof(h_global), h_global, sizeof(h_global), ACL_MEMCPY_HOST_TO_DEVICE));
    h_args[48] = (int64_t)(uintptr_t)d_local;
    h_args[49] = (int64_t)(uintptr_t)d_global;

    void *d_args = nullptr;
    ACL_CHECK(aclrtMalloc(&d_args, kArgsSlots * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_args, kArgsSlots * sizeof(int64_t), h_args.data(),
                          kArgsSlots * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    launch_replay(d_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[replay_host] done: kNumTensors=%d\\n", kNumTensors);

{frees}
    aclrtFree(d_tensors);
    aclrtFree(d_local);
    aclrtFree(d_global);
    aclrtFree(d_args);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}}
"""


def emit_cmakelists(arch: str, name: str, cfg, debug: bool = False) -> str:
    # With -g, also drop the linker `-s` (strip) so the device kernel's
    # debug_line survives -> Insight can map instructions to source lines.
    link_opts = "-Wl,-z,relro -Wl,-z,now" if debug else "-s -Wl,-z,relro -Wl,-z,now"
    dbg_flag = "\n    -g" if debug else ""
    return f"""\
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project(l0_swimlane_{name}_replay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set (source CANN set_env.sh first)")
endif()
set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
set(SOC_VERSION {cfg["soc"]} CACHE STRING "Simulator SoC version")
# Passed by the Python driver as -DPTO_ISA_ROOT= (pin-resolved); never $ENV (#1403).
set(PTO_ISA_ROOT "" CACHE PATH "PTO ISA root")
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR "PTO_ISA_ROOT must be passed as -DPTO_ISA_ROOT=<pin-resolved path>")
endif()
set(REPO_ROOT $ENV{{REPO_ROOT}} CACHE PATH "simpler repo root")

add_compile_options(
    -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong -fPIC
)
add_link_options({link_opts})

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)
set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

set(COMMON_INCLUDES
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/include/pto
    ${{REPO_ROOT}}/src/{arch}/runtime/tensormap_and_ringbuffer/runtime
    ${{REPO_ROOT}}/src/{arch}/runtime/tensormap_and_ringbuffer/common
    ${{REPO_ROOT}}/src/common/task_interface
    ${{REPO_ROOT}}/src/{arch}/platform/include
    ${{REPO_ROOT}}/simpler_setup/incore
    ${{ASCEND_HOME_PATH}}/pkg_inc
    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling
    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
    ${{ASCEND_HOME_PATH}}/include
)

add_library(replay_kernel SHARED replay_kernel.cpp replay_launch.cpp)
target_compile_options(replay_kernel PRIVATE
    ${{CMAKE_CCE_COMPILE_OPTIONS}}
    --cce-aicore-arch={cfg["aicore_arch"]}
    -DREGISTER_BASE -std=c++17{dbg_flag})
target_include_directories(replay_kernel PRIVATE ${{COMMON_INCLUDES}})
target_link_options(replay_kernel PRIVATE --cce-fatobj-link)

add_executable(replay_host replay_host.cpp)
target_compile_options(replay_host PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories(replay_host PRIVATE ${{COMMON_INCLUDES}})
target_link_directories(replay_host PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
    ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib
)
target_link_libraries(replay_host PRIVATE
    replay_kernel
    runtime_camodel
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
"""


def emit_run_collect(cfg, pto_isa_root: str) -> str:
    # Plain string (bash uses ${} braces) — bake SoC default and the
    # pin-resolved pto-isa path via tokens (no ambient PTO_ISA_ROOT env #1403).
    return _RUN_COLLECT_TEMPLATE.replace("__SOC_DEFAULT__", cfg["soc"]).replace("__PTO_ISA_ROOT__", pto_isa_root)


_RUN_COLLECT_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
: "${CANN_HOME:?CANN_HOME must be set}"
: "${REPO_ROOT:?REPO_ROOT must be set}"

WS="${WS:-$(dirname "$(readlink -f "$0")")}"
SOC_VERSION="${SOC_VERSION:-__SOC_DEFAULT__}"
PTO_ISA_ROOT="__PTO_ISA_ROOT__"
DEVICE_ID="${TARGET_DEVICE_ID:-${NPU_LOCKED_DEVICE:-0}}"
BUILD_DIR="$WS/build"
COLLECT_DIR="$WS/msprof_collect"
EXPORT_ROOT="$WS/insight_export"

source "$CANN_HOME/set_env.sh"
export ASCEND_HOME_PATH="$CANN_HOME"
SIM_LIB_DIR="$CANN_HOME/aarch64-linux/simulator/$SOC_VERSION/lib"
LD_LIBS="$BUILD_DIR:$SIM_LIB_DIR:$CANN_HOME/lib64"
LD_LIBS="$LD_LIBS:$CANN_HOME/aarch64-linux/devlib:$CANN_HOME/devlib"
export LD_LIBRARY_PATH="$LD_LIBS:${LD_LIBRARY_PATH:-}"
export ACL_DEVICE_ID="$DEVICE_ID"
mkdir -p "$BUILD_DIR" "$COLLECT_DIR" "$EXPORT_ROOT"

cmake -G Ninja -S "$WS" -B "$BUILD_DIR" \\
    -DSOC_VERSION="$SOC_VERSION" -DPTO_ISA_ROOT="$PTO_ISA_ROOT" -DREPO_ROOT="$REPO_ROOT"
cmake --build "$BUILD_DIR" --target replay_host

msprof op simulator \\
    --application="$BUILD_DIR/replay_host" --kernel-name="replay_entry" \\
    --launch-count=1 --soc-version="$SOC_VERSION" --timeout=120 \\
    --output="$COLLECT_DIR/out" 2>&1 | tee "$COLLECT_DIR/msprof_collect.log"

OPPROF_DIR="$(find "$COLLECT_DIR/out" -maxdepth 1 -mindepth 1 -type d -name 'OPPROF_*' | sort | tail -n 1)"
test -n "$OPPROF_DIR"
if [[ -d "$OPPROF_DIR/device0/tmp_dump" ]]; then
    EXPORT_SRC="$OPPROF_DIR/device0/tmp_dump"
else
    EXPORT_SRC="$OPPROF_DIR/dump"
fi
msprof op simulator --export="$EXPORT_SRC" --output="$EXPORT_ROOT" \\
    2>&1 | tee "$EXPORT_ROOT/msprof_export.log"
echo "[run_collect] done. Insight artifacts under: $EXPORT_ROOT/OPPROF_*/simulator/"
"""


def generate_workspace(  # noqa: PLR0913
    ws: Path,
    arch: str,
    cfg,
    members,
    name: str,
    tensor_count: int,
    args,
    pto_isa_root: str,
    debug: bool = False,
    block_num: int = 1,
):
    ws.mkdir(parents=True, exist_ok=True)
    # One combined replay_entry for the whole mix task (cube=AIC member, vec=AIV
    # member), driven by ONE shared args[] and a single-block `<<<1>>>` launch
    # (one block = 1 AIC + 2 AIV sub-cores). A single-kernel task is just a mix
    # of size 1.
    (ws / "replay_kernel.cpp").write_text(emit_replay_kernel_combined(members, cfg))
    (ws / "replay_launch.cpp").write_text(emit_replay_launch())
    (ws / "replay_host.cpp").write_text(emit_replay_host(tensor_count, args, block_num))
    (ws / "CMakeLists.txt").write_text(emit_cmakelists(arch, name, cfg, debug))
    rc = ws / "run_collect.sh"
    rc.write_text(emit_run_collect(cfg, pto_isa_root))
    rc.chmod(0o755)


# ---------------------------------------------------------------------------
# Step 5: build + collect
# ---------------------------------------------------------------------------
def _build_env():
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann:
        raise OSError("ASCEND_HOME_PATH is not set — source CANN set_env.sh first")
    env = dict(os.environ)
    env["ASCEND_HOME_PATH"] = cann
    env["CANN_HOME"] = cann
    env["REPO_ROOT"] = str(PROJECT_ROOT)
    return env


def smoke_build(ws: Path, env, cfg, pto_isa_root: str):
    build = ws / "build"
    build.mkdir(exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(ws),
            "-B",
            str(build),
            f"-DSOC_VERSION={cfg['soc']}",
            f"-DPTO_ISA_ROOT={pto_isa_root}",
            f"-DREPO_ROOT={env['REPO_ROOT']}",
        ],
        cwd=str(ws),
        env=env,
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "replay_host"], cwd=str(ws), env=env, check=True)
    so = build / "libreplay_kernel.so"
    out = subprocess.run(["nm", "-D", str(so)], check=False, capture_output=True, text=True).stdout
    syms = {line.split()[-1] for line in out.splitlines() if " T " in line}
    for s in ("replay_entry", "launch_replay"):
        if s not in syms:
            raise RuntimeError(f"smoke build: missing symbol {s} in {so}")
    print("[l0_swimlane] smoke build OK (replay_entry, launch_replay present)")


def _to_perfetto(d):  # noqa: PLR0912
    """In-place transform of an Insight trace into a Perfetto-friendly one.

    Insight encodes each pipe as one track and packs concurrent, pipelined
    instructions onto it; Perfetto drops overlapping `ph:X` complete events and
    can mis-pair `B`/`E` flag events. This fixes both, losslessly (same events,
    same ts/dur; only `tid` changes and `B`/`E` is re-encoded as `X`):

      * sub-lanes: greedily pack each duration event into the lowest lane whose
        previous event ended, so no two events on a lane overlap. A split pipe
        `MTE1` becomes `MTE1#0..#k`.
      * atomic flags: merge each `B`/`E` pair into one `ph:X` slice.
      * thread_sort_index rebuilt as base*100+lane so lanes render adjacent.
    """
    EPS = 1e-9
    evs = d["traceEvents"]
    orig_sort = {(e["pid"], e["tid"]): e["args"]["sort_index"] for e in evs if e.get("name") == "thread_sort_index"}

    def is_core(e):
        return str(e.get("pid", "")).startswith("core")

    # 1. Duration intervals per (pid,tid): X events + B/E pairs (matched by id).
    intervals = defaultdict(list)
    be = defaultdict(dict)
    for e in evs:
        if not is_core(e):
            continue
        ph = e.get("ph")
        key = (e["pid"], e["tid"])
        if ph == "X":
            intervals[key].append(
                {
                    "ts": e["ts"],
                    "end": e["ts"] + e.get("dur", 0.0),
                    "name": e["name"],
                    "args": e.get("args", {}),
                }
            )
        elif ph in ("B", "E"):
            slot = be[(e["pid"], e["tid"], e.get("id"))]
            slot[ph] = e["ts"]
            slot.setdefault("src", e)
    for (pid, tid, eid), slot in be.items():
        s = slot.get("B", slot.get("E"))
        en = slot.get("E", slot.get("B"))
        if s is not None and en is not None and s > en:
            s, en = en, s
        src = slot["src"]
        intervals[(pid, tid)].append(
            {
                "ts": s,
                "end": en,
                "name": src["name"],
                "args": src.get("args", {}),
                "id": eid,
            }
        )

    # 2. Greedy lane assignment (interval partitioning) per track.
    max_lane = defaultdict(int)
    be_lane = {}
    for key, iv in intervals.items():
        iv.sort(key=lambda t: (t["ts"], -(t["end"] - t["ts"])))
        lane_end = []
        for it in iv:
            placed = None
            for ln in range(len(lane_end)):
                if lane_end[ln] <= it["ts"] + EPS:
                    placed = ln
                    break
            if placed is None:
                placed = len(lane_end)
                lane_end.append(it["end"])
            else:
                lane_end[placed] = it["end"]
            it["lane"] = placed
            max_lane[key] = max(max_lane[key], placed)
            if "id" in it:
                be_lane[(key[0], key[1], it["id"])] = placed

    def split(pid, tid):
        return max_lane.get((pid, tid), 0) > 0

    def laned(pid, tid, lane):
        return f"{tid}#{lane}" if split(pid, tid) else tid

    out = []
    # 3. Emit every duration interval as an atomic X with its sub-lane tid.
    for (pid, tid), iv in intervals.items():
        for it in iv:
            out.append(
                {
                    "ph": "X",
                    "name": it["name"],
                    "ts": it["ts"],
                    "dur": it["end"] - it["ts"],
                    "pid": pid,
                    "tid": laned(pid, tid, it["lane"]),
                    "args": it["args"],
                }
            )

    # 4. Pass through the rest; anchor flow arrows to their id-pair's lane.
    for e in evs:
        ph = e.get("ph")
        if e.get("name") == "thread_sort_index" or ph in ("X", "B", "E"):
            continue
        ne = dict(e)
        if is_core(e) and ph in ("s", "t", "f") and split(e["pid"], e["tid"]):
            ln = be_lane.get((e["pid"], e["tid"], e.get("id")), 0)
            ne["tid"] = f"{e['tid']}#{ln}"
        out.append(ne)

    # 5. Rebuild thread_sort_index: integer, contiguous, lanes adjacent.
    for (pid, tid), base in orig_sort.items():
        mx = max_lane.get((pid, tid), 0)
        if mx > 0:
            for k in range(mx + 1):
                out.append(
                    {
                        "args": {"sort_index": base * 100 + k},
                        "name": "thread_sort_index",
                        "ph": "M",
                        "pid": pid,
                        "tid": f"{tid}#{k}",
                    }
                )
        else:
            out.append(
                {"args": {"sort_index": base * 100}, "name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid}
            )

    d["traceEvents"] = out
    return d


def collect(ws: Path, env, max_time: int, device=None, dest_name: str = "trace.json"):
    if device:
        # Already inside an outer task-submit lock (the recommended onboard
        # workflow: wrap the whole tool so the dump and this collect share one
        # device — `device` is the appended --device <id>). Reuse it for the
        # camodel collect rather than nesting a second task-submit, which would
        # grab a different device and could deadlock against the outer lock.
        cmd = ["bash", str(ws / "run_collect.sh")]
        env = {**env, "TARGET_DEVICE_ID": device}
    elif shutil.which("task-submit") is not None:
        # Standalone (no outer lock): self-lock just the collect step.
        cmd = [
            "task-submit",
            "--device",
            "auto",
            "--max-time",
            str(max_time),
            "--run",
            f"CANN_HOME={env['CANN_HOME']} REPO_ROOT={env['REPO_ROOT']} "
            f"TARGET_DEVICE_ID=$TASK_DEVICE "
            f"bash {ws}/run_collect.sh",
        ]
    else:
        print(
            "[l0_swimlane] WARNING: task-submit not found; running run_collect.sh "
            "unlocked (results may be noisy if another process shares the NPU)"
        )
        cmd = ["bash", str(ws / "run_collect.sh")]
        env = {**env, "TARGET_DEVICE_ID": os.environ.get("ACL_DEVICE_ID", "0")}
    subprocess.run(cmd, cwd=str(ws), env=env, check=True)

    sims = list(ws.glob("insight_export/OPPROF_*/simulator"))
    if not sims:
        raise RuntimeError("no simulator/ export produced — check msprof_collect.log")
    trace = sorted(sims, key=lambda p: p.stat().st_mtime)[-1] / "trace.json"
    if not trace.is_file():
        raise RuntimeError(f"trace.json missing under {trace.parent}")
    dst = ws / dest_name
    perfetto_dst = ws / dest_name.replace(".json", "_perfetto.json")
    # Pretty-print the Insight copy (one record per line); the export original
    # stays compact (Insight loads the simulator/ directory). Then emit a
    # Perfetto-friendly version (sub-lanes + atomic flags). Fall back to a
    # verbatim copy if the JSON can't be parsed.
    try:
        with open(trace) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _to_perfetto(data)  # mutates data in place (after the pretty dump above)
        with open(perfetto_dst, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, OSError):
        shutil.copy(trace, dst)
        perfetto_dst = None
    return dst, perfetto_dst


# ---------------------------------------------------------------------------
def apply_arg_overrides(kargs: list[dict], set_arg, ap: argparse.ArgumentParser):
    """Apply --set-arg SLOT=VALUE overrides to reconstructed args.

    A scalar slot has its value rewritten; a tensor slot has its data buffer
    filled with VALUE (every element) instead of memset-0. Both shrink a replay
    loop count without touching shapes: single-task kernels carry the count as a
    scalar (n_blocks), the mix paged-attention kernel derives it from the
    context_lens tensor's content. Only real arg slots from the dump are
    settable; tensor fill requires an integer dtype. Increasing a scalar beyond
    its dump value risks out-of-bounds and is warned about.
    """
    if not set_arg:
        return
    by_slot = {a["slot"]: a for a in kargs}
    for spec in set_arg:
        if "=" not in spec:
            ap.error(f"--set-arg must be SLOT=VALUE (got {spec!r})")
        slot_s, val_s = spec.split("=", 1)
        try:
            slot, value = int(slot_s), int(val_s)
        except ValueError:
            ap.error(f"--set-arg SLOT and VALUE must be integers (got {spec!r})")
        a = by_slot.get(slot)
        if a is None:
            ap.error(f"--set-arg slot {slot} is not an arg of this task")
        if a["kind"] == "scalar":
            old = a["value"]
            a["value"] = value
            note = "  WARNING: larger than dump value -> buffers may be undersized" if value > old else ""
            print(f"[l0_swimlane] overriding scalar slot {slot}: {old} -> {value}{note}")
        else:
            if a["dtype"] not in INTEGER_DTYPES:
                ap.error(
                    f"--set-arg slot {slot} is a {a['dtype']} tensor; tensor "
                    f"fill is only supported for integer dtypes (loop-count / "
                    f"index control tensors like context_lens)"
                )
            a["fill"] = value
            print(f"[l0_swimlane] filling tensor slot {slot} ({a['dtype']} {a['shape']}) with {value} in every element")


def main():
    ap = argparse.ArgumentParser(
        description="Generate an AICore intra-core swimlane trace.json for one kernel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ----- what to trace -----
    ap.add_argument("--test", required=True, help="SceneTest test file (.py)")
    ap.add_argument(
        "--func-id",
        required=True,
        metavar="A[,B,C]",
        help="mix member set: comma-separated func_ids of the kernels that form "
        "the task. Name the task's FULL set — `--func-id 0` for a task the "
        "orchestration dispatches on its own, `--func-id 0,1,2` for a 3-way mix "
        "(all its members). The set must exactly match a dispatched task's "
        "func_id array (you wrote the orchestration, so you know the members).",
    )
    ap.add_argument("--task-id", default=None, help="task_id hex (default: lowest)")
    # ----- dump source -----
    ap.add_argument(
        "--platform",
        default="a2a3sim",
        help="dump platform (default a2a3sim). Sim variants "
        "(a2a3sim/a5sim) need no NPU. Onboard variants "
        "(a2a3/a5) run the dump on a real device — wrap the "
        "whole tool in task-submit so the dump and collect "
        "share the locked $TASK_DEVICE. Required for kernels "
        "whose sync idiom (e.g. manual prod.record()) only "
        "compiles for the device, not the cpu sim.",
    )
    ap.add_argument(
        "--device",
        default=None,
        metavar="ID",
        help="NPU device id for an onboard dump + collect. Normally "
        "supplied automatically — task-submit appends "
        "--device <id> to the wrapped command; also read from "
        "$TASK_DEVICE. Sim platforms ignore it.",
    )
    ap.add_argument(
        "--case",
        default=None,
        metavar="NAME",
        help="pin the dump to one CASES[*].name (e.g. SmallCase1). "
        "Omitting it auto-pins the FIRST case that lists --platform "
        "(deterministic dump). Pass it explicitly when that first "
        "case is a full-size production case whose shapes overflow "
        "the camodel replay — name the small one instead. Accepts "
        "ClassName::Case too.",
    )
    ap.add_argument("--dump-json", default=None, help="reuse an existing args_dump.json")
    # ----- replay tuning -----
    ap.add_argument(
        "--set-arg",
        action="append",
        default=[],
        metavar="SLOT=VALUE",
        help="override an arg by args[] slot for the replay. Scalar "
        "slot -> rewrite its value; tensor slot -> fill its data "
        "buffer with VALUE (integer dtypes only). Use to shrink a "
        "loop count for the camodel without distorting pipeline "
        "structure: single-task n_blocks is a scalar "
        "(--set-arg 4=4); mix paged-attention derives n_blocks "
        "from the context_lens tensor (--set-arg 4=512 -> "
        "n_blocks=ceil(512/block_size)). Only real arg slots are "
        "settable. Repeatable. Default: real dump values.",
    )
    ap.add_argument(
        "--spmd-block-num",
        type=int,
        default=None,
        metavar="N",
        help="block_num written into the synthesized SPMD LocalContext "
        "(slot 48). Default: the case's block_dim. Only matters for "
        "kernels that branch/stride on block_num; set the real grid "
        "width for those.",
    )
    ap.add_argument(
        "--debug-line",
        "-g",
        action="store_true",
        help="compile the kernel with -g (and skip link strip) so the "
        "trace carries debug_line -> Insight maps instructions to "
        "source lines. Default off.",
    )
    # ----- run control -----
    ap.add_argument("--no-collect", action="store_true", help="smoke build only")
    ap.add_argument("--max-time", type=int, default=1800, help="task-submit budget (sec)")
    args = ap.parse_args()

    test_path = Path(args.test).resolve()
    arch, variant = parse_platform(args.platform)
    cfg = ARCH_CONFIG.get(arch)
    if cfg is None:
        ap.error(f"unsupported arch {arch} (from {args.platform}); supported: {', '.join(ARCH_CONFIG)}")

    func_id_list = [int(x) for x in args.func_id.split(",")]
    meta = load_kernel_meta(test_path, func_id_list[0], args.platform)
    by_func = meta["by_func"]
    name, class_name = meta["name"], meta["class_name"]

    # Which case the dump runs. Explicit --case wins; otherwise auto-pin the
    # first case that lists --platform so the dump targets exactly one case
    # (deterministic — no "run all default cases, reconstruct from the newest
    # dump dir" ambiguity). Still None only when the test declares no case on
    # this platform (a single-case / no-CASES test — the dump then runs as-is).
    selected_case = args.case or meta["auto_case"]
    if not args.case and meta["auto_case"]:
        print(f"[l0_swimlane] no --case; auto-pinned first {args.platform} case: {meta['auto_case']}")
    # block_num for the synthesized slot-48 LocalContext: the grid width of the
    # SELECTED case (--case bare name, ignoring any ClassName:: prefix), not an
    # arbitrary CASES entry. Defaults to 1 (a non-SPMD single block) when the
    # selected case declares no block_dim, or when no case is selected (a
    # single-case / no-CASES test) — never guessed from a different case.
    case_key = selected_case.split("::")[-1] if selected_case else None
    block_dim = meta["block_dim_by_case"].get(case_key, 1)
    block_num = args.spmd_block_num if args.spmd_block_num is not None else block_dim

    # task-submit hands the locked device by appending --device <id> to argv
    # (and may also set $TASK_DEVICE). One resolved value threads through both
    # the dump and the collect so they share the single outer lock.
    device = args.device or os.environ.get("TASK_DEVICE")
    manifest = get_or_run_dump(test_path, args.platform, variant, args.dump_json, selected_case, device)
    print(f"[l0_swimlane] manifest: {manifest}")

    # Select the task whose member set == --func-id; reconstruct its full
    # positional payload. mix_func_ids is the dump's array (slot order
    # AIC,AIV0,AIV1), NOT the typed order, so lane assignment stays correct.
    chosen, tensor_count, kargs, mix_func_ids = reconstruct_task_args(manifest, func_id_list, args.task_id)

    # Resolve the mix members (slot order) to their sources/core_types.
    missing = [f for f in mix_func_ids if f not in by_func]
    if missing:
        ap.error(f"dump task has func_id(s) {missing} with no matching incore in {test_path.name}")
    members = [by_func[f] for f in mix_func_ids]
    for m in members:
        if m["core_type"] not in ("aic", "aiv"):
            ap.error(f"unsupported core_type {m['core_type']} for func_id {m['func_id']} (only aic/aiv)")
    mode = "mix" if len(members) > 1 else members[0]["core_type"]
    member_desc = ", ".join(f"{m['name']}({m['core_type']},func {m['func_id']})" for m in members)
    print(
        f"[l0_swimlane] func_id={func_id_list} task={chosen} mix={mix_func_ids} mode={mode} "
        f"block_dim={block_dim}\n              members=[{member_desc}]"
    )

    scalars = [a for a in kargs if a["kind"] == "scalar"]
    print(f"[l0_swimlane] task {chosen}: {tensor_count} tensors, {len(scalars)} scalars")
    # Full arg-slot map so the caller can pick a slot for --set-arg without
    # cross-referencing the kernel source. Names are not in the dump (only
    # kind/shape/value) — read the kernel's `args:` header for those.
    print("[l0_swimlane] arg slots (override with --set-arg SLOT=VALUE):")
    for a in sorted(kargs, key=lambda x: x["slot"]):
        if a["kind"] == "tensor":
            print(f"    slot {a['slot']:<2} tensor  {a['dtype']:<8} {a['shape']}")
        else:
            print(f"    slot {a['slot']:<2} scalar  = {a['value']}")
    apply_arg_overrides(kargs, args.set_arg, ap)

    # Self-describing label: <TestClass>_<Case>_<platform>_<kernel>_<mix>, so the
    # workspace dir and the trace.json filename say which case/kernel/mix they are.
    case = _case_from_manifest(manifest, class_name)
    mix_tag = "_".join(str(f) for f in mix_func_ids)
    label = f"{class_name}_{case}_{args.platform}_{name}_mix{mix_tag}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ws = PROJECT_ROOT / "outputs" / f"l0_swimlane_{label}_{ts}"
    # Pin-resolved checkout, threaded by value into the workspace generator (bakes
    # it into run_collect.sh) and smoke_build (-DPTO_ISA_ROOT=). Never exported
    # into the subprocess environment — CMakeLists does not read $ENV (#1403).
    pto_isa_root = ensure_pto_isa_root(verbose=True)
    generate_workspace(
        ws,
        arch,
        cfg,
        members,
        name,
        tensor_count,
        kargs,
        pto_isa_root,
        debug=args.debug_line,
        block_num=block_num,
    )
    print(f"[l0_swimlane] workspace: {ws}")

    env = _build_env()
    smoke_build(ws, env, cfg, pto_isa_root)
    if args.no_collect:
        print(f"[l0_swimlane] --no-collect: stopping after smoke build.\n  {ws}")
        return
    trace, trace_perfetto = collect(ws, env, args.max_time, device, dest_name=f"{label}_trace.json")
    print(f"[l0_swimlane] DONE. trace.json (Insight) -> {trace}")
    if trace_perfetto:
        print(f"[l0_swimlane]       perfetto-friendly  -> {trace_perfetto}")


if __name__ == "__main__":
    main()
