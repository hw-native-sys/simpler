#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard benchmark: cold (no prewarm) vs hot (init prewarm) prebuilt_runtime_arena TIMING.

Each scenario runs in a fresh process so the DeviceRunner prebuilt-arena cache does
not carry over between cold and hot measurements.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

from simpler import _log
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from simpler_setup.scene_test import _build_chip_task_args

_HERE = Path(__file__).resolve().parent
if str(_HERE / "dummy_task") not in sys.path:
    sys.path.insert(0, str(_HERE / "dummy_task"))

from test_dummy_task import TestDummyTask  # noqa: E402

_RUNTIME = "tensormap_and_ringbuffer"
_CASE_NAME = "SingleDummyAutoDep"
_RING = 64

_TIMING_RE = re.compile(r"TIMING: prebuilt_runtime_arena = (\d+)ms")
_STRACE_BIND_RE = re.compile(
    r"\[STRACE\].*name=simpler_run\.bind\.prebuilt\s+ts=\d+\s+dur=(\d+)"
)
_STRACE_PREWARM_RE = re.compile(
    r"\[STRACE\].*name=simpler_prewarm\.build\s+ts=\d+\s+dur=(\d+)"
)


def _last_ms_from_ns(values: list[int]) -> float | None:
    if not values:
        return None
    return values[-1] / 1_000_000.0


def _parse_log(text: str) -> dict[str, float | None]:
    timing_ms = [int(m) for m in _TIMING_RE.findall(text)]
    bind_ns = [int(m) for m in _STRACE_BIND_RE.findall(text)]
    prewarm_ns = [int(m) for m in _STRACE_PREWARM_RE.findall(text)]
    bind_ms = _last_ms_from_ns(bind_ns)
    return {
        "prebuilt_runtime_arena_ms": float(timing_ms[-1]) if timing_ms else bind_ms,
        "simpler_prewarm_build_ms": _last_ms_from_ns(prewarm_ns),
        "bind_prebuilt_strace_ms": bind_ms,
    }


def _run_scenario(*, device_id: int, platform: str, prewarm: bool) -> None:
    logging.getLogger("simpler").setLevel(_log.V9)

    case = next(c for c in TestDummyTask.CASES if c["name"] == _CASE_NAME)
    callable_obj = TestDummyTask.compile_chip_callable(platform)

    worker = Worker(
        level=2,
        device_id=device_id,
        platform=platform,
        runtime=_RUNTIME,
    )
    handle = worker.register(callable_obj)

    ring_cfg = CallConfig()
    ring_cfg.runtime_env.ring_task_window = _RING

    try:
        if prewarm:
            worker.init(prewarm_config=ring_cfg)
        else:
            worker.init()

        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = TestDummyTask.CALLABLE.get("orchestration", {}).get("signature", [])

        test_args = TestDummyTask().generate_args(params)
        chip_args, _output_names = _build_chip_task_args(test_args, orch_sig)

        run_cfg = CallConfig()
        run_cfg.block_dim = config_dict.get("block_dim", 1)
        run_cfg.aicpu_thread_num = config_dict.get("aicpu_thread_num", 2)
        run_cfg.runtime_env.ring_task_window = _RING

        worker.run(handle, chip_args, config=run_cfg)
    finally:
        worker.close()


def _child_main(mode: str, device_id: int, platform: str) -> int:
    _run_scenario(device_id=device_id, platform=platform, prewarm=(mode == "hot"))
    return 0


def _spawn(mode: str, device_id: int, platform: str) -> tuple[str, dict[str, float | None]]:
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            mode,
            "--device",
            str(device_id),
            "--platform",
            platform,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log = proc.stdout or ""
    if proc.returncode != 0:
        print(log, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return log, _parse_log(log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--mode", choices=["cold", "hot", "both"], default="both")
    parser.add_argument("--child", choices=["cold", "hot"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        raise SystemExit(_child_main(args.child, args.device, args.platform))

    modes = ["cold", "hot"] if args.mode == "both" else [args.mode]
    results: dict[str, dict[str, float | None]] = {}

    for mode in modes:
        print(f"=== {mode.upper()} (prewarm={'yes' if mode == 'hot' else 'no'}) ===", flush=True)
        log, parsed = _spawn(mode, args.device, args.platform)
        results[mode] = parsed
        for line in log.splitlines():
            if any(
                token in line
                for token in (
                    "TIMING: prebuilt_runtime_arena",
                    "name=simpler_prewarm.build",
                    "name=simpler_run.bind.prebuilt",
                )
            ):
                print(line)
        prewarm_val = parsed["simpler_prewarm_build_ms"]
        prewarm_text = f"{prewarm_val:.2f}ms" if prewarm_val is not None else "None"
        print(
            f"  -> bind.prebuilt(STRACE)={parsed['bind_prebuilt_strace_ms']:.2f}ms"
            f", prewarm.build={prewarm_text}",
            flush=True,
        )

    if args.mode == "both":
        cold = results["cold"]["bind_prebuilt_strace_ms"]
        hot = results["hot"]["bind_prebuilt_strace_ms"]
        prewarm_ms = results["hot"]["simpler_prewarm_build_ms"]
        print("\n=== SUMMARY ===")
        print(f"cold first-run bind.prebuilt (STRACE) : {cold:.2f} ms")
        print(f"hot  first-run bind.prebuilt (STRACE) : {hot:.2f} ms")
        print(f"hot  init simpler_prewarm.build        : {prewarm_ms:.2f} ms")
        if cold is not None and hot is not None:
            print(f"first-run savings (cold - hot)         : {cold - hot:.2f} ms")
            if prewarm_ms is not None:
                print(f"(build cost shifted to init prewarm) : ~{prewarm_ms:.2f} ms")


if __name__ == "__main__":
    main()
