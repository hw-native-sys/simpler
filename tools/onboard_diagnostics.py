# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Temporary onboard CI evidence capture; the pytest command's exit status is preserved."""

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def sample_occupancy(destination, stop):
    with destination.open("w") as output:
        while not stop.is_set():
            device_pids = set()
            output.write(f"\nTIME {datetime.now(timezone.utc).isoformat()}\n")
            for command in (["task-submit", "--list"], ["npu-smi", "info"], ["ps", "-eo", "pid,ppid,user,lstart,comm"]):
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
                    lines = result.stdout.splitlines()
                    if command[0] == "task-submit":
                        for index, line in enumerate(lines):
                            task = re.search(r"task_[0-9_]+", line)
                            if task:
                                device = re.search(r"\[NPU:[^]]+\]", line)
                                lines[index] = task[0] + (" " + device[0] if device else "")
                    elif command[0] == "npu-smi":
                        device_pids = {
                            int(pid) for pid in re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", result.stdout)
                        }
                    else:
                        records = {
                            int(parts[0]): (int(parts[1]), line)
                            for line in lines
                            for parts in [line.split()]
                            if len(parts) > 2 and parts[0].isdigit() and parts[1].isdigit()
                        }
                        selected = set(device_pids)
                        for pid in device_pids:
                            while pid in records and records[pid][0] not in selected:
                                pid = records[pid][0]
                                selected.add(pid)
                        lines = [records[pid][1] for pid in sorted(selected) if pid in records]
                    output.write(
                        f"COMMAND {command!r} rc={result.returncode}\n" + "\n".join(lines) + "\n" + result.stderr
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    output.write(f"QUERY ERROR {error}\n")
            output.flush()
            stop.wait(2)


@contextlib.contextmanager
def capture_case():
    started = time.time()
    output = Path(tempfile.mkdtemp(prefix="simpler-onboard-evidence-"))
    ascend = output / "ascend"
    ascend.mkdir()
    previous = os.environ.get("ASCEND_PROCESS_LOG_PATH")
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(ascend)
    stop = threading.Event()
    monitor = threading.Thread(target=sample_occupancy, args=(output / "occupancy.log", stop), daemon=True)
    monitor.start()
    try:
        yield
    finally:
        stop.set()
        monitor.join(timeout=30)
        pids = {os.getpid()}
        module = sys.modules.get("simpler.task_interface")
        if module is not None:
            module._native_flush_host_log(1000)
            directory = module._native_host_log_directory() or module._HOST_LOG_SESSION_DIRECTORY
            for source in Path(directory).glob("host.*.log"):
                pids.add(int(source.name.split(".")[1]))
                shutil.copy2(source, output / source.name)
                print(
                    f"\n[HOST EVIDENCE] {source}\n" + "\n".join(source.read_text(errors="replace").splitlines()[-120:])
                )
        collect_default_device_logs(ascend, pids, started)
        print_device_errors(ascend)
        print("\n[OCCUPANCY EVIDENCE]\n" + (output / "occupancy.log").read_text())
        if previous is None:
            os.environ.pop("ASCEND_PROCESS_LOG_PATH", None)
        else:
            os.environ["ASCEND_PROCESS_LOG_PATH"] = previous


def collect_default_device_logs(destination, pids, started):
    for root_id, root in enumerate((Path.home() / "ascend/log", Path("/root/ascend/log"))):
        for kind in ("debug", "run"):
            for pid in pids:
                for pattern in (f"plog/plog-{pid}_*.log", f"device-*/device-{pid}_*.log"):
                    for source in (root / kind).glob(pattern):
                        try:
                            if source.stat().st_mtime >= started - 30:
                                target = destination / f"{root_id}-{kind}-{source.parent.name}-{source.name}"
                                shutil.copy2(source, target)
                        except OSError as error:
                            print(f"[EVIDENCE COLLECTION ERROR] {source}: {error}")


def print_device_errors(directory):
    for source in sorted(directory.rglob("*.log")):
        matches = [
            line
            for line in source.read_text(errors="replace").splitlines()
            if any(word in line for word in ("[ERROR]", "PrintCoreInfo", "HandleTaskTimeout", "sub_class="))
        ]
        if matches:
            print(f"\n[DEVICE EVIDENCE] {source}", flush=True)
            print("\n".join(matches[-40:]), flush=True)
