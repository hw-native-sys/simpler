# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reserve both network1 hosts, supervise their work, and retain cleanup evidence."""

import argparse
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from reservation import event, write_json

_HELPER = ".github/actions/network1-run-pytest/reservation.py"
_QUEUE_WAIT_S = None  # use the job's remaining budget; tests can set a shorter limit
_CLEANUP_WAIT_S = 120
_POLL_S = 0.2
_JOB_SECONDS = 3600
_FINALIZE_MARGIN_S = 120
_RPC_TIMEOUT_S = 30
_PREFLIGHT_S = 30  # npu-smi and ps, each with a 15-second limit
_MIN_PYTEST_S = 120  # leave enough time to run a useful smoke after a long queue wait
_QUEUE_OVERRUN_S = 2 * _RPC_TIMEOUT_S  # status and receipt reads may finish after the queue deadline
_WORK_RPCS = 4  # a final readiness/result poll can overrun its phase deadline
_TASK_ID = re.compile(r"task_[0-9_]+\Z")


class Host:
    def __init__(self, journal, record):
        self.journal = journal
        self.record = record
        self.state = Path(record["state"])
        self.helper = str(Path(record["cwd"]) / _HELPER)

    def run(self, argv, data=None, check=True, timeout=_RPC_TIMEOUT_S):
        command = argv
        if self.record["remote"]:
            command = [
                "ssh",
                "-p",
                os.environ["NETWORK1_REMOTE_SSH_PORT"],
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                os.environ["REMOTE_TARGET"],
                shlex.join(argv),
            ]
        result = subprocess.run(command, input=data, text=True, capture_output=True, timeout=timeout, check=False)
        with (self.journal / f"control-{self.record['machine']}.log").open("a") as output:
            output.write(
                f"{time.time():.6f} {shlex.join(argv)}\n{result.stdout}{result.stderr}\nexit={result.returncode}\n"
            )
        if check and result.returncode:
            raise RuntimeError(f"host {self.record['machine']}: {shlex.join(argv)} failed: {result.stderr}")
        return result

    def control(self, operation, data=None):
        return self.run(["python3", self.helper, operation, str(self.state)], data=data)

    def read(self):
        return json.loads(self.control("read").stdout)

    def status(self):
        value = self.run(["task-submit", "--status", self.record["task_id"]]).stdout.strip()
        if not re.fullmatch(r"pending|running|not_found|completed \(exit=-?\d+\)", value):
            raise RuntimeError(f"unrecognized task status: {value!r}")
        return value

    def recover_submission(self):
        if self.record.get("intent") and not self.record.get("task_id"):
            found = self.run(["task-submit", "--find", self.record["key"]]).stdout.splitlines()
            found = [value for value in found if _TASK_ID.fullmatch(value)]
            if len(found) == 1:
                self.record["task_id"] = found[0]
            elif found:
                raise RuntimeError(f"ambiguous owned submission: {found}")
            else:
                state = self.read()
                receipt = state.get("granted") or state.get("rejected") or state.get("closed", {})
                if _TASK_ID.fullmatch(receipt.get("task_id", "")):
                    self.record["task_id"] = receipt["task_id"]
                else:
                    raise RuntimeError("submission acknowledgement lost; ownership remains unknown")


def save(journal, records):
    write_json(journal / "tasks.json", records)


def _body(cann, command, environment):
    exports = "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in environment.items())
    mkdir = '\nmkdir -p "$ASCEND_PROCESS_LOG_PATH"' if "ASCEND_PROCESS_LOG_PATH" in environment else ""
    return [
        "bash",
        "-c",
        f"set -eo pipefail\nsource {shlex.quote(cann)}\nsource .venv/bin/activate\n"
        f"{exports}{mkdir}\nexec {shlex.join(command)}",
    ]


def configure(journal, env):
    started = float((Path(env["RUNNER_TEMP"]) / "network1-job-start").read_text())
    local: dict = dict(
        machine=env["NETWORK1_MACHINE"],
        cwd=env["GITHUB_WORKSPACE"],
        devices=env["NETWORK1_LOCAL_DEVICES"],
        remote=False,
        state=str(journal / "local"),
    )
    remote: dict = dict(
        machine=env["NETWORK1_REMOTE_MACHINE"],
        cwd=env["REMOTE_WORKDIR"],
        devices=env["NETWORK1_REMOTE_DEVICES"],
        remote=True,
        state=str(Path(env["REMOTE_WORKDIR"]) / "output/network1-ci/reservation"),
    )
    if local["machine"] == remote["machine"]:
        raise ValueError("network1 requires distinct machine identities")
    records = sorted([local, remote], key=lambda record: record["machine"])
    watch = {
        key: env[key]
        for key in ("SIMPLER_SCHEDULER_TIMEOUT_MS", "SIMPLER_OP_EXECUTE_TIMEOUT_US", "SIMPLER_STREAM_SYNC_TIMEOUT_MS")
    }
    for record in records:
        record["key"] = f"network1-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}-{uuid.uuid4().hex}"
    save(journal, records)
    for record in records:
        host = Host(journal, record)
        host.run(["task-submit", "--version"])
        host.run(["task-submit", "--list"])
        if record["remote"]:
            host.run(["mkdir", "-p", str(Path(record["cwd"]) / "output/network1-ci/pytest")])
    remaining = _JOB_SECONDS - (time.time() - started)
    # Queue as long as possible while leaving a minimum smoke and proven cleanup.
    # The actual pytest timeout is recalculated after both grants arrive.
    minimum_work = math.ceil(
        int(env["NETWORK1_DAEMON_WAIT_S"])
        + min(_MIN_PYTEST_S, int(env["NETWORK1_SMOKE_TIMEOUT_S"]))
        + 2 * _CLEANUP_WAIT_S
        + _PREFLIGHT_S
        + (_WORK_RPCS + 2) * _RPC_TIMEOUT_S  # daemon and pytest start controls
        + 2 * _POLL_S
    )
    configure_seconds = len(records) * _RPC_TIMEOUT_S
    # Both hosts share one queue deadline. The first holder may wait for the
    # second for that entire period, so its watchdog includes the same wait.
    queue_seconds = int(remaining - minimum_work - _FINALIZE_MARGIN_S - configure_seconds - _QUEUE_OVERRUN_S)
    if _QUEUE_WAIT_S is not None:
        queue_seconds = min(_QUEUE_WAIT_S, queue_seconds)
    if queue_seconds < 1:
        raise TimeoutError("insufficient network1 job budget for queue, run and cleanup")
    # Configuration runs before task-submit starts the holder's max-time clock.
    hold_seconds = queue_seconds + _QUEUE_OVERRUN_S + minimum_work + configure_seconds
    event(
        journal,
        "budget",
        remaining=remaining,
        queue_seconds=queue_seconds,
        hold_seconds=hold_seconds,
        setup_seconds=configure_seconds,
        preflight_seconds=_PREFLIGHT_S,
        minimum_pytest_seconds=min(_MIN_PYTEST_S, int(env["NETWORK1_SMOKE_TIMEOUT_S"])),
        queue_overrun_seconds=_QUEUE_OVERRUN_S,
    )
    for record in records:
        record["queue_seconds"] = queue_seconds
        host = Host(journal, record)
        if record["remote"]:
            role = "daemon"
            environment = {
                **watch,
                "PYTHONPATH": "",
                "CMAKE_PREFIX_PATH": "",
                "ASCEND_PROCESS_LOG_PATH": str(
                    Path(record["cwd"]) / "output/network1-ci/pytest" / f"daemon-{record['machine']}"
                ),
            }
            command = [
                "python",
                "-m",
                "simpler.remote_l3_worker",
                "--host",
                env["REMOTE_DAEMON_HOST"],
                "--port",
                env["NETWORK1_L3_DAEMON_PORT"],
            ]
            cann = env["NETWORK1_REMOTE_CANN_ENV"]
        else:
            role = "pytest"
            environment = {
                **watch,
                "NETWORK1_REMOTE_ENDPOINT": f"{env['REMOTE_DAEMON_HOST']}:{env['NETWORK1_L3_DAEMON_PORT']}",
            }
            command = ["python", "-m", "pytest", *shlex.split(env["PYTEST_ARGS"])]
            cann = env["NETWORK1_CANN_ENV"]
        record["hold_seconds"] = hold_seconds
        config: dict = dict(
            role=role,
            devices=record["devices"],
            cwd=record["cwd"],
            hold_seconds=hold_seconds,
            cleanup_seconds=_CLEANUP_WAIT_S,
            admission_seconds=queue_seconds + configure_seconds,
            command=_body(cann, command, environment),
        )
        if record["remote"]:
            config["endpoint"] = [env["REMOTE_DAEMON_HOST"], int(env["NETWORK1_L3_DAEMON_PORT"])]
        host.control("configure", json.dumps(config))
        save(journal, records)
    return records


def acquire(journal, records):
    deadline = time.monotonic() + records[0]["queue_seconds"]
    for record in records:
        host = Host(journal, record)
        command = shlex.join(["python3", host.helper, "worker", record["state"], record["key"]])
        record["intent"] = True
        save(journal, records)
        event(journal, "queued", machine=record["machine"])
        output = host.run(
            ["task-submit", "--device", record["devices"], "--max-time", str(record["hold_seconds"]), command]
        ).stdout.splitlines()
        identities = [value for value in output if _TASK_ID.fullmatch(value)]
        if len(identities) != 1:
            raise RuntimeError(f"submission did not return one task identity: {output}")
        record["task_id"] = identities[0]
        save(journal, records)
        while time.monotonic() < deadline:
            status = host.status()
            state = host.read()
            if state.get("rejected", {}).get("task_id") == record["task_id"]:
                raise RuntimeError(state["rejected"]["reason"])
            if "drained" in state:
                raise RuntimeError("reservation worker drained before both hosts passed preflight; see before.log")
            if (
                status == "running"
                and state.get("granted", {}).get("task_id") == record["task_id"]
                and state.get("preflight", {}).get("task_id") == record["task_id"]
            ):
                event(journal, "granted", machine=record["machine"], task_id=record["task_id"])
                break
            if status not in ("pending", "running"):
                raise RuntimeError(f"reservation exited before grant: {status}")
            time.sleep(_POLL_S)
        else:
            raise TimeoutError("two-host reservation queue wait expired")


def pytest_budget(env):
    started = float((Path(env["RUNNER_TEMP"]) / "network1-job-start").read_text())
    remaining = _JOB_SECONDS - (time.time() - started)
    # A late grant shortens pytest, never the time needed to drain and release.
    reserve = 2 * _CLEANUP_WAIT_S + _FINALIZE_MARGIN_S + _WORK_RPCS * _RPC_TIMEOUT_S
    seconds = min(int(env["NETWORK1_SMOKE_TIMEOUT_S"]), int(remaining - reserve))
    if seconds < 1:
        raise TimeoutError("insufficient network1 job budget for pytest and cleanup")
    return seconds


def run_work(journal, records, env):
    remote = Host(journal, next(record for record in records if record["remote"]))
    local = Host(journal, next(record for record in records if not record["remote"]))
    remote.control("start")
    event(journal, "daemon_start")
    deadline = time.monotonic() + int(env["NETWORK1_DAEMON_WAIT_S"])
    while time.monotonic() < deadline:
        state = remote.read()
        if any(key in state for key in ("result", "drained")) or remote.status() != "running":
            raise RuntimeError("peer daemon exited before readiness; see retained work.log")
        if state.get("ready", {}).get("task_id") != remote.record["task_id"]:
            time.sleep(_POLL_S)
            continue
        try:
            with socket.create_connection((env["REMOTE_DAEMON_HOST"], int(env["NETWORK1_L3_DAEMON_PORT"])), timeout=1):
                break
        except OSError:
            time.sleep(_POLL_S)
    else:
        raise TimeoutError("peer daemon readiness expired after reservation grant")
    local.control("start")
    smoke_seconds = pytest_budget(env)
    event(journal, "pytest_start", timeout_seconds=smoke_seconds)
    deadline = time.monotonic() + smoke_seconds
    while time.monotonic() < deadline:
        if any(key in remote.read() for key in ("result", "drained")) or remote.status() != "running":
            raise RuntimeError("peer daemon exited while pytest was running")
        state = local.read()
        if "result" in state:
            event(journal, "pytest_finish", exit_code=state["result"]["exit_code"])
            return state["result"]["exit_code"]
        if "drained" in state:
            raise RuntimeError("local worker drained before reporting pytest result")
        if local.status() != "running":
            raise RuntimeError("local task exited without pytest result")
        time.sleep(_POLL_S)
    raise TimeoutError("network1 pytest execution deadline expired")


def request_stop(host):
    host.recover_submission()
    status = host.status()
    if status == "pending":
        cancelled = host.run(["task-submit", "--cancel", host.record["task_id"]], check=False)
        host.record["pending_cancel_ack"] = cancelled.returncode == 0
        status = host.status()
    if status == "running":
        host.control("stop")


def await_drained(host, deadline):
    record = host.record
    host.recover_submission()
    while True:
        status = host.status()
        state = host.read()
        cancelled = status == "not_found" and record.get("pending_cancel_ack") and "granted" not in state
        if cancelled:
            found = host.run(["task-submit", "--find", record["key"]]).stdout.strip()
            cancelled = not found and host.status() == "not_found"
        closed = status.startswith("completed ") and state.get("closed", {}).get("task_id") == record["task_id"]
        drained = status == "running" and state.get("drained", {}).get("task_id") == record["task_id"]
        if cancelled or closed or drained:
            record["drained"] = drained
            record["retired"] = cancelled or closed
            record.pop("cleanup_error", None)
            return
        if time.monotonic() >= deadline:
            # --kill is a request, never evidence of terminal ownership.
            host.run(["task-submit", "--kill", record["task_id"]], check=False)
            raise TimeoutError(f"reservation retirement unproved: {status}")
        request_stop(host)
        time.sleep(_POLL_S)


def release_pair(journal, records):
    complete = True
    for record in records:
        if record.get("intent") and not record.get("retired"):
            try:
                Host(journal, record).control("release")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                record["cleanup_error"] = str(error)
                complete = False
    deadline = time.monotonic() + _CLEANUP_WAIT_S
    for record in records:
        if not record.get("intent") or record.get("retired"):
            continue
        host = Host(journal, record)
        try:
            while time.monotonic() < deadline:
                status = host.status()
                state = host.read()
                if status.startswith("completed ") and state.get("closed", {}).get("task_id") == record["task_id"]:
                    record["retired"] = True
                    break
                time.sleep(_POLL_S)
            else:
                raise TimeoutError("released reservation has not closed and reached terminal status")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            record["cleanup_error"] = str(error)
            complete = False
        save(journal, records)
    return complete


def collect_queue_state(journal, records, deadline):
    for record in records:
        if not record.get("intent"):
            continue
        for option in ("--version", "--list"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                event(journal, "queue_snapshot_unavailable", machine=record["machine"], reason="cleanup budget")
                return
            try:
                result = Host(journal, record).run(
                    ["task-submit", option], check=False, timeout=min(_RPC_TIMEOUT_S, remaining)
                )
                event(journal, "queue_snapshot", machine=record["machine"], option=option, exit_code=result.returncode)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                event(journal, "queue_snapshot_unavailable", machine=record["machine"], option=option, error=str(error))


def collect_artifacts(journal, records):
    for record in records:
        if not record.get("task_id"):
            continue
        try:
            result = Host(journal, record).run(["task-submit", "--log", record["task_id"]], check=False)
            print(f"network1 host {record['machine']} task {record['task_id']}\n{result.stdout}", flush=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            event(journal, "task_log_unavailable", machine=record["machine"], error=str(error))
    remote = next(record for record in records if record["remote"])
    complete = True
    try:
        for source, destination in (
            ("output/network1-ci/pytest/", journal.parent / "pytest"),
            ("output/network1-ci/reservation/", journal / "peer"),
        ):
            destination.mkdir(parents=True, exist_ok=True)
            ssh = shlex.join(
                [
                    "ssh",
                    "-p",
                    os.environ["NETWORK1_REMOTE_SSH_PORT"],
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ConnectTimeout=10",
                ]
            )
            result = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "-e",
                    ssh,
                    f"{os.environ['REMOTE_TARGET']}:{remote['cwd']}/{source}",
                    str(destination) + "/",
                ],
                timeout=30,
                check=False,
            )
            if result.returncode:
                complete = False
    except (OSError, subprocess.TimeoutExpired) as error:
        event(journal, "artifact_copy_failed", error=str(error))
        complete = False
    return complete


def cleanup(journal, records):
    event(journal, "cleanup_start")
    for record in records:
        if not record.get("intent"):
            continue
        try:
            request_stop(Host(journal, record))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            record["cleanup_error"] = str(error)
            event(journal, "stop_request_failed", machine=record["machine"], error=str(error))
        save(journal, records)
    deadline = time.monotonic() + _CLEANUP_WAIT_S
    for record in records:
        if not record.get("intent"):
            continue
        try:
            await_drained(Host(journal, record), deadline)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            record["cleanup_error"] = str(error)
        save(journal, records)
    complete = all(
        not record.get("intent")
        or ((record.get("retired") or record.get("drained")) and not record.get("cleanup_error"))
        for record in records
    )
    if complete:
        event(journal, "pair_drained")
        collect_queue_state(journal, records, deadline)
        complete = release_pair(journal, records)
    artifacts_copied = collect_artifacts(journal, records) if any(record.get("intent") for record in records) else True
    complete = complete and artifacts_copied
    remote = next(record for record in records if record["remote"])
    write_json(journal / "cleanup.json", dict(complete=complete, remote_workdir=remote["cwd"], records=records))
    event(journal, "cleanup_finish", complete=complete)
    if not complete:
        print(f"network1 ownership/cleanup is unproved; retain {remote['cwd']} and {journal}", file=sys.stderr)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "cleanup"))
    args = parser.parse_args()
    journal = Path(os.environ["RUN_DIR"]) / "reservations"
    journal.mkdir(parents=True, exist_ok=True)
    if args.operation == "cleanup":
        if not (journal / "tasks.json").exists():
            return 0
        try:
            if json.loads((journal / "cleanup.json").read_text()).get("complete"):
                return 0
        except (OSError, ValueError):
            pass
        return 0 if cleanup(journal, json.loads((journal / "tasks.json").read_text())) else 1

    def interrupted(signum, _):
        raise InterruptedError(f"network1 coordinator received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    result = 1
    try:
        records = configure(journal, os.environ)
        acquire(journal, records)
        result = run_work(journal, records, os.environ)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        event(journal, "failed", error=str(error))
        print(error, file=sys.stderr)
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if (journal / "tasks.json").exists():
            records = json.loads((journal / "tasks.json").read_text())
            if not cleanup(journal, records):
                result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
