# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side lifetime of one network1 CI device reservation (Linux only)."""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_TASK_ID = re.compile(r"task_[0-9_]+\Z")
_PROC = Path("/proc")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def event(state, name, **fields):
    with (state / "timeline.jsonl").open("a") as output:
        output.write(json.dumps(dict(event=name, time=time.time(), **fields)) + "\n")


@contextmanager
def gate(state, exclusive):
    with (state / "gate").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def process_info(pid):
    fields = (_PROC / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    return fields[0], int(fields[1]), fields[19]


def process_uid(path):
    for line in (path / "status").read_text().splitlines():
        if line.startswith("Uid:"):
            return int(line.split()[2])
    raise RuntimeError(f"effective UID is unavailable for {path.name}")


def live_grant(state):
    grant = json.loads((state / "granted.json").read_text())
    status, _, start = process_info(grant["pid"])
    if status == "Z" or start != grant["start"] or not _TASK_ID.fullmatch(grant["task_id"]):
        raise RuntimeError("reservation owner is not live")
    return grant


def control(state, command):
    with gate(state, exclusive=True):
        if command == "start" and (state / "closing").exists():
            raise RuntimeError("reservation is closing")
        if command == "release" and not (state / "drained.json").exists():
            raise RuntimeError("reservation has not drained")
        (state / ("closing" if command == "stop" else command)).touch()
        event(state, command)


def rank(state, argv):
    # Closing takes the exclusive gate before scanning accepted processes.
    with gate(state, exclusive=False):
        grant = live_grant(state)
        if grant["uid"] not in (0, os.geteuid()):
            raise RuntimeError("MPI rank UID is outside the reservation scanner's ownership domain")
        if not (state / "start").exists() or (state / "closing").exists():
            raise RuntimeError("MPI rank has no active reservation")
        env = {**os.environ, **grant["environment"]}
        # MPI supplies PMI communication through an inherited descriptor.
        # The rank launcher must keep it open when it starts the test Python.
        child = subprocess.Popen(argv, env=env, close_fds=False)
    return child.wait()


def snapshot(state, name, deadline=None):
    successful = True
    with (state / f"{name}.log").open("w") as output:
        for command in (
            ["npu-smi", "info"],
            ["ps", "-eo", "pid,ppid,user,lstart,args"],
        ):
            output.write(f"\n{time.time():.6f} {json.dumps(command)}\n")
            output.flush()
            try:
                seconds = 15 if deadline is None else min(15, deadline - time.monotonic())
                if seconds <= 0:
                    raise TimeoutError("diagnostic cleanup budget expired")
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=seconds, check=False)
                output.write(f"exit={result.returncode}\n")
                successful &= result.returncode == 0
            except (OSError, subprocess.TimeoutExpired) as error:
                output.write(f"query failed: {error}\n")
                successful = False
    return successful


def marked_processes(task_id, excluded, since):
    marker = f"TASKQUEUE_TASK_ID={task_id}".encode()
    found = []
    for path in _PROC.glob("[0-9]*"):
        pid = int(path.name)
        if pid in excluded:
            continue
        try:
            status, _, start = process_info(pid)
            if status == "Z" or int(start) < int(since):
                continue
            if os.geteuid() and process_uid(path) != os.geteuid():
                continue
            if marker in (path / "environ").read_bytes().split(b"\0"):
                found.append((pid, start))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            raise RuntimeError(f"process ownership is unreadable for PID {pid}") from error
    return found


def signal_owned(pid, start, sig=signal.SIGTERM):
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if process_info(pid)[2] != start:
            return
        # The scanner already proved the marker. The stable handle and birth
        # check preserve that identity even if environ becomes unreadable at exit.
        signal.pidfd_send_signal(descriptor, sig)
    except (FileNotFoundError, ProcessLookupError):
        return
    finally:
        os.close(descriptor)


def check_process_handles():
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("network1 reservations require Linux pidfd_open and pidfd_send_signal")
    descriptor = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)


def owns_listener(pid, address, port):
    inodes = set()
    for path in (_PROC / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    encoded = socket.inet_aton(socket.gethostbyname(address))[::-1].hex().upper()
    endpoint = f"{encoded}:{port:04X}"
    for line in (_PROC / str(pid) / "net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == endpoint and fields[3] == "0A" and fields[9] in inodes:
            return True
    return False


def read_state(state):
    values = {
        name: json.loads(path.read_text())
        for name in ("granted", "rejected", "preflight", "result", "ready", "drained", "closed")
        if (path := state / f"{name}.json").exists()
    }
    if "ready" in values and (
        (state / "closing").exists() or any(key in values for key in ("result", "drained", "closed"))
    ):
        values.pop("ready")
    if "ready" in values:
        ready = values["ready"]
        try:
            holder = live_grant(state)
            status, _, start = process_info(ready["pid"])
            if status == "Z" or start != ready["start"] or ready["task_id"] != holder["task_id"]:
                values.pop("ready")
        except (FileNotFoundError, ProcessLookupError, RuntimeError):
            values.pop("ready")
    return values


def publish_ready(state, child, config, task_id):
    if child is None or config["role"] != "daemon" or (state / "ready.json").exists():
        return
    address, port = config["endpoint"]
    if owns_listener(child.pid, address, port):
        marker = f"TASKQUEUE_TASK_ID={task_id}".encode()
        if marker not in (_PROC / str(child.pid) / "environ").read_bytes().split(b"\0"):
            raise RuntimeError("managed listener does not belong to this reservation")
        write_json(
            state / "ready.json",
            dict(task_id=task_id, pid=child.pid, start=process_info(child.pid)[2], address=address, port=port),
        )
        event(state, "ready", pid=child.pid)


def wait_for_stop(state, started, seconds, stopped):
    while not stopped() and not (state / "closing").exists():
        if time.monotonic() - started >= seconds:
            raise TimeoutError("reservation lifetime expired during cleanup wait")
        time.sleep(0.1)


def drain(state, task_id, seconds):
    excluded = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        excluded.add(parent)
        try:
            _, parent, _ = process_info(parent)
        except (OSError, ValueError, IndexError):
            break
    started = time.monotonic()
    deadline = started + seconds
    escalate_at = started + min(5, seconds / 2)
    since = process_info(os.getpid())[2]
    while True:
        try:
            remaining = marked_processes(task_id, excluded, since)
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise
            event(state, "drain_scan_unknown", error=str(error))
            time.sleep(0.1)
            continue
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"owned processes did not drain: {remaining}")
        sig = signal.SIGKILL if time.monotonic() >= escalate_at else signal.SIGTERM
        for pid, start in remaining:
            signal_owned(pid, start, sig)
        event(state, "draining", pids=[pid for pid, _ in remaining], signal=sig.name)
        time.sleep(0.1)


def publish_preflight(state, task_id, stopped):
    with gate(state, exclusive=True):
        if stopped() or (state / "closing").exists():
            return
        write_json(state / "preflight.json", dict(task_id=task_id, time=time.time()))
        event(state, "preflight_complete", task_id=task_id)


def worker(state):
    config = json.loads((state / "config.json").read_text())
    task_id = os.environ.get("TASKQUEUE_TASK_ID", "")
    if not _TASK_ID.fullmatch(task_id) or os.environ.get("TASK_DEVICE") != config["devices"]:
        raise RuntimeError("task-submit did not grant the requested devices and task identity")
    now = time.monotonic()
    if (
        config["boot_id"] != (_PROC / "sys/kernel/random/boot_id").read_text().strip()
        or not config["admission_started"] <= now < config["admission_expires"]
    ):
        rejection = dict(task_id=task_id, reason="reservation admission expired or host rebooted", time=time.time())
        write_json(state / "rejected.json", rejection)
        write_json(state / "closed.json", rejection)
        event(state, "admission_rejected", task_id=task_id)
        return 1
    stop_requested = False

    def stop(*_):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    started = time.monotonic()
    child = None
    result = 1
    try:
        check_process_handles()
        event(state, "process_handles_ready")
        with gate(state, exclusive=True):
            if (state / "closing").exists():
                return 1
            environment = {
                key: os.environ[key]
                for key in ("TASKQUEUE_TASK_ID", "TASKQUEUE_INSIDE", "TASK_DEVICE", "NPU_LOCKED_DEVICE")
                if key in os.environ
            }
            write_json(
                state / "granted.json",
                dict(
                    task_id=task_id,
                    pid=os.getpid(),
                    uid=os.geteuid(),
                    start=process_info(os.getpid())[2],
                    environment=environment,
                ),
            )
        event(state, "granted", task_id=task_id, devices=config["devices"])
        if not snapshot(state, "before"):
            raise RuntimeError("reservation preflight queries failed; see before.log")
        publish_preflight(state, task_id, lambda: stop_requested)
        while not stop_requested and not (state / "closing").exists():
            if time.monotonic() - started >= config["hold_seconds"]:
                raise TimeoutError("reservation lifetime expired")
            if child is None and (state / "start").exists():
                with gate(state, exclusive=False):
                    if not (state / "closing").exists():
                        output = (state / "work.log").open("w")
                        child = subprocess.Popen(
                            config["command"], cwd=config["cwd"], stdout=output, stderr=subprocess.STDOUT
                        )
                        output.close()
                        event(state, "work_started", pid=child.pid)
            if child is not None and child.poll() is not None:
                result = child.returncode
                write_json(state / "result.json", dict(exit_code=result))
                event(state, "work_finished", exit_code=result)
                if config["role"] == "daemon":
                    break
                # The local reservation also covers peer teardown and diagnostics.
                wait_for_stop(state, started, config["hold_seconds"], lambda: stop_requested)
                break
            publish_ready(state, child, config, task_id)
            time.sleep(0.1)
    finally:
        control(state, "stop")
        cleanup_deadline = time.monotonic() + config["cleanup_seconds"]
        drain(state, task_id, max(0, cleanup_deadline - time.monotonic()))
        if child is not None:
            child.wait()
        snapshot(state, "after", deadline=cleanup_deadline)
        drain(state, task_id, max(0, cleanup_deadline - time.monotonic()))
        write_json(state / "drained.json", dict(task_id=task_id, time=time.time()))
        event(state, "drained", task_id=task_id)
        while not (state / "release").exists():
            if time.monotonic() - started >= config["hold_seconds"] + config["cleanup_seconds"]:
                raise TimeoutError("pair-wide release unproved; reservation lifetime expired")
            time.sleep(0.1)
        write_json(state / "closed.json", dict(task_id=task_id, time=time.time()))
        event(state, "closed", task_id=task_id)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("configure", "read", "start", "stop", "release", "worker", "rank"))
    parser.add_argument("state", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == "configure":
        args.state.mkdir(parents=True, exist_ok=True)
        config = json.load(sys.stdin)
        config["boot_id"] = (_PROC / "sys/kernel/random/boot_id").read_text().strip()
        config["admission_started"] = time.monotonic()
        config["admission_expires"] = config["admission_started"] + config.pop("admission_seconds")
        with (args.state / "config.json").open("x") as output:
            json.dump(config, output)
        (args.state / "gate").touch()
    elif args.operation == "read":
        print(json.dumps(read_state(args.state)))
    elif args.operation in ("start", "stop", "release"):
        control(args.state, args.operation)
    elif args.operation == "worker":
        return worker(args.state)
    else:
        return rank(args.state, args.arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
