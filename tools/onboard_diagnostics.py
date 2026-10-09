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
        # Two separate obligations, and the environment is the one that must
        # survive the other failing. Evidence collection reads directories this
        # user may not own -- the fallback roots below are another user's on a
        # shared host -- and an error raised out of here would both skip the
        # restore and be reported by pytest as a teardown error, turning a
        # passing or skipped case into a failing one. The case's own exception,
        # if it had one, is still in flight and is never replaced: nothing here
        # raises.
        try:
            stop.set()
            monitor.join(timeout=30)
            collect_evidence(output, ascend, started)
        except Exception as error:  # noqa: BLE001 -- evidence is never the verdict
            print(f"[EVIDENCE COLLECTION ERROR] {type(error).__name__}: {error}", flush=True)
        finally:
            if previous is None:
                os.environ.pop("ASCEND_PROCESS_LOG_PATH", None)
            else:
                os.environ["ASCEND_PROCESS_LOG_PATH"] = previous


def collect_evidence(output, ascend, started):
    """Copy this case's logs beside its artifacts and echo a bounded summary.

    Every file read here is best-effort: a source that has gone away, or that
    belongs to another user, is reported and skipped. The full text stays in
    the copied files; only the console view is truncated, because the job log
    is what a reader scrolls and what the runner has to upload.
    """
    pids = {os.getpid()}
    module = sys.modules.get("simpler.task_interface")
    if module is not None:
        module._native_flush_host_log(1000)
        directory = module._native_host_log_directory() or module._HOST_LOG_SESSION_DIRECTORY
        for source in safe_iterdir(Path(directory), "host.*.log"):
            try:
                pids.add(int(source.name.split(".")[1]))
                shutil.copy2(source, output / source.name)
                print(f"\n[HOST EVIDENCE] {source}\n" + tail_lines(source, 120))
            except (OSError, ValueError, IndexError) as error:
                print(f"[EVIDENCE COLLECTION ERROR] {source}: {error}")
    collect_default_device_logs(ascend, pids, started)
    print_device_errors(ascend)
    occupancy = output / "occupancy.log"
    print(f"\n[OCCUPANCY EVIDENCE] {occupancy}\n" + tail_lines(occupancy, 200), flush=True)


def safe_iterdir(base, pattern):
    """`base.glob(pattern)` as a list, with an unreadable or absent `base` empty.

    `Path.glob` is lazy, so a directory the caller may not enter raises on the
    *first advance* of the generator rather than at the call -- which is why a
    `try` around the loop body alone does not contain it.
    """
    try:
        return sorted(base.glob(pattern))
    except OSError as error:
        print(f"[EVIDENCE COLLECTION ERROR] {base}/{pattern}: {error}")
        return []


def tail_lines(source, limit):
    """The last `limit` lines of `source`, or a note saying why there are none."""
    try:
        lines = source.read_text(errors="replace").splitlines()
    except OSError as error:
        return f"[EVIDENCE COLLECTION ERROR] {source}: {error}"
    dropped = len(lines) - limit
    head = f"[{dropped} earlier line(s) omitted; the full file is at {source}]\n" if dropped > 0 else ""
    return head + "\n".join(lines[-limit:])


def collect_default_device_logs(destination, pids, started):
    for root_id, root in enumerate((Path.home() / "ascend/log", Path("/root/ascend/log"))):
        for kind in ("debug", "run"):
            for pid in pids:
                for pattern in (f"plog/plog-{pid}_*.log", f"device-*/device-{pid}_*.log"):
                    for source in safe_iterdir(root / kind, pattern):
                        try:
                            if source.stat().st_mtime >= started - 30:
                                target = destination / f"{root_id}-{kind}-{source.parent.name}-{source.name}"
                                shutil.copy2(source, target)
                        except OSError as error:
                            print(f"[EVIDENCE COLLECTION ERROR] {source}: {error}")


def print_device_errors(directory):
    for source in safe_rglob(directory, "*.log"):
        try:
            lines = source.read_text(errors="replace").splitlines()
        except OSError as error:
            print(f"[EVIDENCE COLLECTION ERROR] {source}: {error}")
            continue
        matches = [
            line
            for line in lines
            if any(word in line for word in ("[ERROR]", "PrintCoreInfo", "HandleTaskTimeout", "sub_class="))
        ]
        if matches:
            print(f"\n[DEVICE EVIDENCE] {source}", flush=True)
            print("\n".join(matches[-40:]), flush=True)


def safe_rglob(base, pattern):
    """`base.rglob(pattern)` as a sorted list; see `safe_iterdir` for the lazy-raise."""
    try:
        return sorted(base.rglob(pattern))
    except OSError as error:
        print(f"[EVIDENCE COLLECTION ERROR] {base}/**/{pattern}: {error}")
        return []
