#!/usr/bin/env python3
"""
Per-rank Python worker for distributed (multi-card) kernel execution.

Replaces the monolithic C++ distributed_worker binary.  Each rank runs
as a separate process, using the comm_* C API (via ctypes bindings) for
HCCL / sim communication and the existing PTO runtime C API for kernel
execution.

Spawned by DistributedCodeRunner — not intended for direct invocation.
"""

import argparse
import struct
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "python"))
sys.path.insert(0, str(script_dir))


DTYPE_FORMAT = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint32": ("I", 4),
    "uint64": ("Q", 8),
    "float16": ("e", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
}


def parse_buffer_spec(spec):
    parts = spec.split(":")
    return {"name": parts[0], "dtype": parts[1], "count": int(parts[2])}


def parse_kernel_spec(spec):
    p = spec.index(":")
    return {"func_id": int(spec[:p]), "filename": spec[p + 1:]}


def main():
    parser = argparse.ArgumentParser(description="Distributed per-rank worker")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nranks", type=int, required=True)
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--rootinfo-file", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--orch-file", required=True)
    parser.add_argument("--orch-func", required=True)
    parser.add_argument("--win-sync-prefix", type=int, default=0)
    parser.add_argument("--aicpu-thread-num", type=int, default=1)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--orch-thread-num", type=int, default=0)
    parser.add_argument("--win-buffer", action="append", default=[])
    parser.add_argument("--dev-buffer", action="append", default=[])
    parser.add_argument("--load", action="append", default=[], dest="loads")
    parser.add_argument("--save", action="append", default=[], dest="saves")
    parser.add_argument("--arg", action="append", default=[], dest="args")
    parser.add_argument("--kernel-bin", action="append", default=[])
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    data_dir = Path(args.data_dir) if args.data_dir else artifact_dir / f"rank_{args.rank}"

    buffers = []
    for spec in args.win_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "window"
        buffers.append(b)
    for spec in args.dev_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "device"
        buffers.append(b)

    kernel_bins = [parse_kernel_spec(s) for s in args.kernel_bin]

    buf_by_name = {b["name"]: b for b in buffers}

    def elem_size(dtype):
        return DTYPE_FORMAT.get(dtype, ("f", 4))[1]

    def buf_bytes(b):
        return b["count"] * elem_size(b["dtype"])

    # ----------------------------------------------------------------
    # 1. Load library
    # ----------------------------------------------------------------
    from bindings import (
        bind_host_binary, set_device, launch_runtime,
        device_malloc, copy_to_device, copy_from_device,
        comm_init, comm_alloc_windows, comm_get_local_window_base,
        comm_barrier, comm_destroy,
    )

    lib_path = artifact_dir / "libhost_runtime.so"
    Runtime = bind_host_binary(str(lib_path))
    set_device(args.device_id)

    sys.stderr.write(f"[rank {args.rank}] Library loaded, device {args.device_id} set\n")

    # ----------------------------------------------------------------
    # 2. Comm init + alloc windows
    # ----------------------------------------------------------------
    comm = comm_init(args.rank, args.nranks, args.rootinfo_file)

    total_win = args.win_sync_prefix
    for b in buffers:
        if b["placement"] == "window":
            total_win += buf_bytes(b)

    device_ctx_ptr = comm_alloc_windows(comm, total_win)
    local_base = comm_get_local_window_base(comm)

    sys.stderr.write(f"[rank {args.rank}] Comm initialized, local_base=0x{local_base:x}\n")

    # ----------------------------------------------------------------
    # 3. Allocate buffers
    # ----------------------------------------------------------------
    win_offset = args.win_sync_prefix

    for b in buffers:
        nbytes = buf_bytes(b)
        if b["placement"] == "window":
            b["dev_ptr"] = local_base + win_offset
            win_offset += nbytes
        else:
            ptr = device_malloc(nbytes)
            if not ptr:
                sys.stderr.write(f"[rank {args.rank}] device_malloc failed for '{b['name']}'\n")
                return 3
            b["dev_ptr"] = ptr
        sys.stderr.write(
            f"[rank {args.rank}] Buffer '{b['name']}': {b['placement']} "
            f"{b['count']}x{b['dtype']}={nbytes}B @ 0x{b['dev_ptr']:x}\n"
        )

    # ----------------------------------------------------------------
    # 4. Load inputs
    # ----------------------------------------------------------------
    for name in args.loads:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --load: buffer '{name}' not found\n")
            return 1
        path = data_dir / f"{name}.bin"
        host_data = path.read_bytes()
        if len(host_data) != buf_bytes(b):
            sys.stderr.write(
                f"[rank {args.rank}] Size mismatch for '{name}': "
                f"file={len(host_data)}, expected={buf_bytes(b)}\n"
            )
            return 2
        import ctypes
        host_buf = (ctypes.c_uint8 * len(host_data)).from_buffer_copy(host_data)
        copy_to_device(b["dev_ptr"], ctypes.addressof(host_buf), len(host_data))

    # ----------------------------------------------------------------
    # 5. Barrier before kernel execution
    # ----------------------------------------------------------------
    comm_barrier(comm)

    # ----------------------------------------------------------------
    # 6. Run simpler runtime
    # ----------------------------------------------------------------
    orch_binary = (artifact_dir / args.orch_file).read_bytes()
    aicpu_binary = (artifact_dir / "libaicpu_kernel.so").read_bytes()
    aicore_binary = (artifact_dir / "aicore_kernel.o").read_bytes()

    kernel_binaries = []
    for k in kernel_bins:
        data = (artifact_dir / k["filename"]).read_bytes()
        kernel_binaries.append((k["func_id"], data))

    func_args = []
    for tok in args.args:
        if tok == "nranks":
            func_args.append(args.nranks)
        elif tok == "root":
            func_args.append(args.root)
        elif tok == "deviceCtx":
            func_args.append(device_ctx_ptr)
        else:
            b = buf_by_name.get(tok)
            if not b:
                sys.stderr.write(f"[rank {args.rank}] --arg: unknown token '{tok}'\n")
                return 1
            func_args.append(b["dev_ptr"])

    sys.stderr.write(
        f"[rank {args.rank}] Launching kernel: {len(func_args)} args, "
        f"{len(kernel_binaries)} kernels\n"
    )

    runtime = Runtime()
    runtime.initialize(
        orch_binary,
        args.orch_func,
        func_args,
        kernel_binaries=kernel_binaries,
    )

    launch_runtime(
        runtime,
        aicpu_thread_num=args.aicpu_thread_num,
        block_dim=args.block_dim,
        device_id=args.device_id,
        aicpu_binary=aicpu_binary,
        aicore_binary=aicore_binary,
        orch_thread_num=args.orch_thread_num,
    )

    runtime.finalize()
    sys.stderr.write(f"[rank {args.rank}] Kernel execution complete\n")

    # ----------------------------------------------------------------
    # 7. Barrier + save outputs
    # ----------------------------------------------------------------
    comm_barrier(comm)

    import ctypes
    for name in args.saves:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --save: buffer '{name}' not found\n")
            continue
        nbytes = buf_bytes(b)
        host_buf = (ctypes.c_uint8 * nbytes)()
        copy_from_device(ctypes.addressof(host_buf), b["dev_ptr"], nbytes)
        path = data_dir / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(host_buf))
        sys.stderr.write(f"[rank {args.rank}] Saved '{name}' to {path} ({nbytes}B)\n")

    # ----------------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------------
    comm_destroy(comm)
    sys.stderr.write(f"[rank {args.rank}] Done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
