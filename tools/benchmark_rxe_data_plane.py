#!/usr/bin/env python3
"""Benchmark gRPC chunking vs RXE data-plane L4/L3 tensor dispatch."""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time
from collections.abc import Iterable

from simpler.distributed.l3_daemon import L3Daemon
from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8192,65536,1048576")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--transports", default="grpc,rxe")
    parser.add_argument("--inline-threshold", type=int, default=4096)
    args = parser.parse_args()

    sizes = [int(item) for item in args.sizes.split(",") if item]
    transports = [item.strip().lower() for item in args.transports.split(",") if item.strip()]

    print("transport,size_bytes,repeats,mean_ms,p50_ms,p95_ms,min_ms,max_ms")
    for transport in transports:
        for size in sizes:
            samples = _benchmark_one(
                transport,
                size,
                repeats=args.repeats,
                warmup=args.warmup,
                inline_threshold=args.inline_threshold,
            )
            print(_format_row(transport, size, samples), flush=True)
    return 0


def _benchmark_one(transport: str, size: int, *, repeats: int, warmup: int, inline_threshold: int) -> list[float]:
    daemon = L3Daemon(0, lambda: Worker(level=3, num_sub_workers=1), tensor_transport=transport)
    endpoint = f"127.0.0.1:{daemon.start()}"
    w4 = Worker(level=4, num_sub_workers=0)

    def l3_orch(orch, args, config):  # noqa: ANN001
        in_tensor = args.tensor(0)
        out_tensor = args.tensor(1)
        data = ctypes.string_at(int(in_tensor.data), int(in_tensor.nbytes()))
        out = bytes(value ^ 0x5A for value in data)
        ctypes.memmove(int(out_tensor.data), out, len(out))

    l3_cid = w4.register(l3_orch)
    w4.add_remote_worker(endpoint, tensor_transport=transport, tensor_inline_threshold=inline_threshold)
    w4.init()

    try:
        samples: list[float] = []
        for iteration in range(warmup + repeats):
            payload = _payload(size, iteration)
            expected = bytes(value ^ 0x5A for value in payload)
            in_buf = ctypes.create_string_buffer(payload, len(payload))
            out_buf = ctypes.create_string_buffer(b"\0" * len(payload), len(payload))

            def l4_orch(orch, args, config):  # noqa: ANN001
                sub_args = TaskArgs()
                sub_args.add_tensor(
                    ContinuousTensor.make(ctypes.addressof(in_buf), (len(payload),), DataType.UINT8),
                    TensorArgType.INPUT,
                )
                sub_args.add_tensor(
                    ContinuousTensor.make(ctypes.addressof(out_buf), (len(payload),), DataType.UINT8),
                    TensorArgType.OUTPUT_EXISTING,
                )
                orch.submit_next_level(l3_cid, sub_args, CallConfig())

            start = time.perf_counter()
            w4.run(l4_orch)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if bytes(out_buf.raw) != expected:
                raise RuntimeError(f"{transport} output mismatch for size={size}, iteration={iteration}")
            if iteration >= warmup:
                samples.append(elapsed_ms)
        return samples
    finally:
        w4.close()
        daemon.stop()


def _payload(size: int, salt: int) -> bytes:
    return bytes((index + salt) % 251 for index in range(size))


def _format_row(transport: str, size: int, samples: Iterable[float]) -> str:
    values = list(samples)
    sorted_values = sorted(values)
    p50 = statistics.median(sorted_values)
    p95 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))]
    return (
        f"{transport},{size},{len(values)},"
        f"{statistics.mean(values):.3f},{p50:.3f},{p95:.3f},{min(values):.3f},{max(values):.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
