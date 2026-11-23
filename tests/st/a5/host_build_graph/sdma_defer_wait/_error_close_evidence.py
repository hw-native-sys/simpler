# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate per-rank SDMA fault snapshots without depending on device-log transport."""

import json
import logging
import os
import subprocess
from itertools import chain


def check_fault_captures(output, devices):
    for rank, device in enumerate(devices):
        paths = list((output / "fault-capture" / f"rank{rank}").rglob("chip_swimlane_records.json"))
        assert len(paths) == 1, f"missing or duplicate fault capture for rank {rank}; evidence: {output}"
        path = paths[0]
        raw = json.loads(path.read_text())
        identity = json.loads(path.with_name("dispatch_identity.json").read_text())
        assert identity["chip_rank"] == rank, f"wrong rank identity in {path}"
        assert identity["run_id"] == 2, f"not the injected run in {path}"
        protocol = raw["aicore_protocol"]
        assert raw["chip_swimlane_level"] == 4 and protocol["schema"] == "hbg-native-protocol", path
        assert protocol["producer"] == "aicore" and protocol["run_epoch"] == 2, path
        # SchedulerGraphResult::ASYNC_WAIT_OVERFLOW is 12 in the native protocol.
        assert protocol["scheduler_error"] == 12, f"unexpected scheduler error in {path}: {protocol}"
        assert protocol["error_claimed"] == 1, path
        tasks = protocol["tasks"]
        assert len(tasks) == 1 and tasks[0]["active_mask"] == 2 and tasks[0]["block_num"] == 1, path
        assert tasks[0]["kernel_ids"][1] == 0, path
        executions = protocol["executions"]
        assert len(executions) == 1, f"missing SDMA kernel execution in {path}"
        execution = executions[0]
        assert execution["task_ref"] == tasks[0]["task_ref"] and execution["run_epoch"] == 2, path
        assert 0 < execution["entry"][1] < execution["returned"][1], f"SDMA kernel did not return in {path}"
        error = protocol["first_error"]
        assert error["task_ref"] == execution["task_ref"] and error["run_epoch"] == 2, path
        assert error["worker_index"] == execution["worker_index"] and error["block_idx"] == 0, path
        controls = protocol["task_controls"]
        assert len(controls) == 1 and controls[0]["task_ref"] == execution["task_ref"], path
        assert controls[0]["state"] != 2 and controls[0]["completed_subtasks"] == 0, (
            f"failed SDMA task reported completion in {path}"
        )
        assert protocol["resolved_task_count"] == 0, f"failed SDMA graph reported completion in {path}"
        (output / f"device-{device}-error.json").write_text(
            json.dumps({"device": device, "identity": identity, "protocol": protocol}, indent=2) + "\n"
        )


def run_error_close_probe(command, output, devices, *, cwd=None):
    output = output.resolve()
    logs = output / "ascend"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        ASCEND_PROCESS_LOG_PATH=str(logs),
        ASCEND_GLOBAL_LOG_LEVEL="3",
        ASCEND_SLOG_PRINT_TO_STDOUT="0",
        ASCEND_LOG_DEVICE_FLUSH_TIMEOUT="10000",
    )
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=180, check=False)
        assert result.returncode == 0, f"SDMA error-close probe exited {result.returncode}; evidence: {output}\n" + (
            output / "stderr.log"
        ).read_text(errors="replace")
        check_fault_captures(output, devices)
        retention = output / "retention.txt"
        assert retention.is_file() and retention.read_text() == (
            f"{len(devices)} ranks: failed SDMA run retained its domain\n"
        ), f"missing or invalid domain-retention proof; evidence: {output}"
    finally:
        # task-submit can run as root; CI uploads artifacts as the runner user.
        try:
            for path in chain((output,), output.rglob("*")):
                try:
                    if not path.is_symlink():
                        path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError as error:
                    logging.warning("SDMA evidence permissions could not be normalized for %s: %s", path, error)
        except OSError as error:
            logging.warning("SDMA evidence traversal failed for %s: %s", output, error)
