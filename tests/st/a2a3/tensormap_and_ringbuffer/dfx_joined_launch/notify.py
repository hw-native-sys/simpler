# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Blocking notifications for the joined-launch test's isolated hardware owner."""

import json
import select
import time


class NotificationInbox:
    """Keep unrelated notifications until the exact waiter consumes them."""

    def __init__(self, channel, *, clock=time.monotonic, poll=select.select):
        self.channel, self.clock, self.poll = channel, clock, poll
        self.pending = []

    def wait(self, kind, timeout, **fields):
        deadline = self.clock() + timeout
        while True:
            for index, event in enumerate(self.pending):
                if event.get("type") == kind and all(event.get(key) == value for key, value in fields.items()):
                    return self.pending.pop(index)
            budget = deadline - self.clock()
            if budget <= 0 or not self.poll([self.channel], [], [], budget)[0]:
                raise TimeoutError(f"notification {kind} for {fields} did not arrive")
            data = self.channel.recv(8192)
            if not data:
                raise RuntimeError("native notification channel closed")
            event = json.loads(data)
            assert isinstance(event, dict), "invalid notification"
            self.pending.append(event)


def build_notify_helper():
    """Compile the test SDK adapter in CPU preparation; device runs reuse this cache."""
    import fcntl  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import os  # noqa: PLC0415
    import platform  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    source = Path(__file__).with_name("notify_gate.cpp")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    pkg = cann / f"{platform.machine()}-linux/pkg_inc"
    includes = [cann / "include", pkg, pkg / "runtime", pkg / "driver", pkg / "profiling"]
    compiler = shutil.which("c++")
    assert compiler is not None, "the Notify test adapter needs a host C++ compiler"
    key = hashlib.sha256(source.read_bytes() + (cann / "include/acl/acl_rt.h").read_bytes())
    key.update(str((cann.resolve(), compiler, platform.machine())).encode())
    directory = source.resolve().parents[5] / "build" / "dfx-notify" / key.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "notify_gate.so"
    with (directory / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            temporary = output.with_suffix(f".{os.getpid()}.so")
            command = [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-pthread", str(source)]
            command += [f"-I{path}" for path in includes]
            command += [
                f"-L{cann / 'lib64'}",
                f"-Wl,-rpath,{cann / 'lib64'}",
                "-lascendcl",
                "-ldl",
                "-o",
                str(temporary),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
            assert result.returncode == 0, f"Notify adapter build failed:\n{result.stdout}\n{result.stderr}"
            temporary.replace(output)
    return output


def check_identity(event, identity):
    """The notification names the exact mailbox dispatch, not only a reused slot."""
    from simpler.worker import _TASK_PROTOCOL_VERSION  # noqa: PLC0415

    fields = (
        "run_id",
        "pipeline_slot",
        "pipeline_generation",
        "endpoint_dispatch_id",
        "task_slot",
        "group_index",
        "group_size",
    )
    expected = [_TASK_PROTOCOL_VERSION, *(identity[key] for key in fields)]
    assert event["pid"] == identity["host_pid"] and event["identity"] == expected, (
        f"notification belongs to a different dispatch: {event}; expected {identity}"
    )


class NotifyGate:
    """One token and one immutable dispatch over an independent SDK control stream."""

    def __init__(self, controller, token):
        self.controller, self.token = controller, token
        self.entered = None
        self.retired = False

    def wait_entered(self, timeout):
        armed = self.controller.wait("armed", timeout, token=self.token)
        self.entered = self.controller.command(f"entered {self.token}", "entered", timeout, token=self.token)
        assert self.entered["identity"] == armed["identity"] and self.entered["pid"] == armed["pid"]

    def verify(self, identity):
        assert self.entered is not None, "gate has no device entered acknowledgement"
        check_identity(self.entered, identity)

    def release(self):
        event = self.controller.command(f"release {self.token}", "released", 10, token=self.token)
        if self.entered is not None:
            assert event["identity"] == self.entered["identity"], "release changed dispatch"

    def close(self):
        if not self.retired:
            self.controller.command(f"retire {self.token}", "retired", 10, token=self.token)
            self.retired = True


class NativeNotifications:
    """Test-scoped activation observer and blocking socket transport; no polling or timing oracle."""

    def __init__(self, prefixes, library):
        import ctypes  # noqa: PLC0415
        import os  # noqa: PLC0415
        import socket  # noqa: PLC0415
        import uuid  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from .process import retain_until_owner_exit  # noqa: PLC0415

        self.gates = {Path(prefix).resolve(): NotifyGate(self, index + 1) for index, prefix in enumerate(prefixes)}
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.settimeout(10)
        name = f"dfx-notify-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.listener.bind("\0" + name)
        self.listener.listen(1)
        self.inbox = None
        self.pids = set()
        self.activations = {}
        self.launches = {}
        self.library = ctypes.CDLL(str(library))
        self.library.joined_notify_configure.argtypes = [ctypes.c_char_p]
        self.library.joined_notify_configure.restype = None
        self.library.joined_notify_configure(name.encode())
        self.library.joined_notify_begin.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        self.library.joined_notify_begin.restype = None
        self.library.joined_notify_end.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.library.joined_notify_end.restype = None
        self.library.joined_notify_launched.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.library.joined_notify_launched.restype = None
        retain_until_owner_exit(self)

    def gate(self, prefix):
        from pathlib import Path  # noqa: PLC0415

        return self.gates[Path(prefix).resolve()]

    def _connection(self):
        if self.inbox is None:
            import socket  # noqa: PLC0415
            import struct  # noqa: PLC0415

            channel = self.listener.accept()[0]
            pid, _uid, _gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            assert pid in self.pids, f"unexpected notification peer {pid}; expected {self.pids}"
            self.inbox = NotificationInbox(channel)
        return self.inbox

    def wait(self, kind, timeout, **fields):
        event = self._connection().wait(kind, timeout, **fields)
        assert event["pid"] in self.pids, "notification changed child"
        return event

    def command(self, command, kind, timeout, **fields):
        inbox = self._connection()
        inbox.channel.sendall(command.encode())
        return self.wait(kind, timeout, **fields)

    def await_activation(self, run_id, timeout):
        if run_id not in self.activations:
            self.activations[run_id] = self.wait("activation", timeout, run_id=run_id)
        return self.activations[run_id]

    def await_launch(self, run_id, timeout):
        if run_id not in self.launches:
            self.launches[run_id] = self.wait("launch", timeout, run_id=run_id)
        return self.launches[run_id]

    def observe(self):
        """Observe real activation/progress transitions; flush native receipts before notifying."""
        import contextlib  # noqa: PLC0415
        import ctypes  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from _task_interface import _flush_host_log  # noqa: PLC0415

        @contextlib.contextmanager
        def installed():
            previous = sys.getprofile()
            launched = set()

            def profile(frame, event, result):
                if previous is not None:
                    previous(frame, event, result)
                name = frame.f_code.co_name
                if frame.f_globals.get("__name__") != "simpler.worker" or name not in (
                    "activate_run",
                    "republish_native_preparation",
                ):
                    return
                staged = frame.f_locals["staged"]
                if name == "activate_run" and event == "call":
                    prefix = Path(staged.config.output_prefix).resolve()
                    matching = [gate for path, gate in self.gates.items() if prefix == path or path in prefix.parents]
                    assert len(matching) <= 1, "ambiguous gate prefix"
                    token = matching[0].token if matching else 0
                    self.library.joined_notify_begin(token, (ctypes.c_uint64 * 8)(*staged.identity))
                elif name == "activate_run" and event == "return":
                    if result is True:
                        disposition = staged.chip_run.preparation_disposition
                        assert _flush_host_log(1000), "activation receipt log flush failed"
                        self.library.joined_notify_end(
                            staged.native_activated,
                            staged.chip_run.launched,
                            int(getattr(disposition, "value", disposition)),
                        )
                    else:
                        self.library.joined_notify_end(0, 0, 0)
                # This real progress checkpoint is also reached after done()
                # retries a deferred launch. Reading launched never drives it.
                if (
                    event == "return"
                    and staged.native_activated
                    and staged.identity not in launched
                    and staged.chip_run.launched
                ):
                    assert _flush_host_log(1000), "launch receipt log flush failed"
                    self.library.joined_notify_launched((ctypes.c_uint64 * 8)(*staged.identity))
                    launched.add(staged.identity)

            sys.setprofile(profile)
            try:
                yield
            finally:
                sys.setprofile(previous)

        return installed()

    def close(self):
        for gate in self.gates.values():
            gate.close()
        if self.inbox is not None:
            self.command("close", "closed", 10)
            self.inbox.channel.close()
        self.listener.close()
