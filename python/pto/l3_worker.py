"""ChipWorker — child process that owns one NPU device.

Each worker process binds to one device via set_device() (DeviceRunner
is a process-global singleton), then loops receiving task specs from the
main process and executing them via the existing bindings.py API.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import sys
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure simpler's python/ is importable
_PYTHON_DIR = str(Path(__file__).resolve().parent.parent)


def _worker_main(device_id: int, artifacts_dir: str,
                 cmd_pipe: multiprocessing.connection.Connection,
                 result_pipe: multiprocessing.connection.Connection):
    """Entry point for a worker process. Runs in a child process."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)

    from bindings import bind_host_binary, set_device, launch_runtime

    # Load host binary and bind to device (once)
    host_binary_path = Path(artifacts_dir) / "host_runtime.so"
    if host_binary_path.exists():
        RuntimeClass = bind_host_binary(str(host_binary_path))
    else:
        # Binary passed inline in first task
        RuntimeClass = None

    set_device(device_id)

    while True:
        try:
            msg = cmd_pipe.recv()
        except EOFError:
            break

        if msg is None:  # shutdown
            break

        task_id, task_spec = msg
        try:
            # Ensure host binary is loaded
            if RuntimeClass is None and "host_binary" in task_spec:
                RuntimeClass = bind_host_binary(task_spec["host_binary"])

            rt = RuntimeClass()
            rt.initialize(
                task_spec["orch_binary"],
                task_spec["orch_func"],
                task_spec["func_args"],
                arg_types=task_spec["arg_types"],
                arg_sizes=task_spec["arg_sizes"],
                kernel_binaries=task_spec["kernel_binaries"],
            )
            launch_runtime(
                rt,
                aicpu_thread_num=task_spec.get("aicpu_thread_num", 1),
                block_dim=task_spec.get("block_dim", 1),
                device_id=device_id,
                aicpu_binary=task_spec["aicpu_binary"],
                aicore_binary=task_spec["aicore_binary"],
                orch_thread_num=task_spec.get("orch_thread_num", 1),
            )
            rt.finalize()
            result_pipe.send((task_id, True, None))
        except Exception as e:
            result_pipe.send((task_id, False, traceback.format_exc()))


class ChipWorker:
    """Manages a child process that owns one NPU device."""

    def __init__(self, device_id: int, artifacts_dir: str = ""):
        self.device_id = device_id
        self.artifacts_dir = artifacts_dir
        self._process: Optional[multiprocessing.Process] = None
        self._cmd_pipe: Optional[multiprocessing.connection.Connection] = None
        self._result_pipe: Optional[multiprocessing.connection.Connection] = None

    def start(self) -> None:
        parent_cmd, child_cmd = multiprocessing.Pipe()
        child_result, parent_result = multiprocessing.Pipe()

        self._cmd_pipe = parent_cmd
        self._result_pipe = parent_result

        self._process = multiprocessing.Process(
            target=_worker_main,
            args=(self.device_id, self.artifacts_dir, child_cmd, child_result),
            daemon=True,
        )
        self._process.start()

    def send_task(self, task_id: int, task_spec: dict) -> None:
        """Send a task to the worker. Non-blocking."""
        self._cmd_pipe.send((task_id, task_spec))

    def poll_result(self, timeout: float = 0.01):
        """Check for a completed result. Returns (task_id, success, error) or None."""
        if self._result_pipe.poll(timeout):
            return self._result_pipe.recv()
        return None

    def stop(self) -> None:
        """Send shutdown signal."""
        if self._cmd_pipe:
            try:
                self._cmd_pipe.send(None)
            except BrokenPipeError:
                pass

    def join(self, timeout: float = 10.0) -> None:
        if self._process:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
