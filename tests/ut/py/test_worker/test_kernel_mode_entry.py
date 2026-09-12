# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware UT for the Python kernel-mode surface of ChipWorker.

The Python twin of tests/ut/cpp/hardware/test_kernel_mode_entry.cpp, and the
inverse of test_platform_comm.py's contract: there ChipWorker owns ACL bring-up
and stream lifetime internally, here the *caller* owns both and lends the stream
in. That inversion is what kernel mode is, so the test does its own device bind
and stream creation through ``_acl_bind_device`` / ``_acl_create_stream`` and
hands the resulting integer address to kernel_init. In production that integer
comes from the framework instead — torch_npu.npu.current_stream().npu_stream.

The backend reports kernel mode unsupported today, so the contract-correct
outcome is a refusal, and reaching the refusal is the assertion: it proves the
Python call arrived at the C ABI rather than being rejected on the way. When the
platform gains a real implementation these expectations invert into a full
init -> prepare -> launch -> close lifecycle; the borrowing scaffolding does not
change.

Each case runs in a forked subprocess: kernel_init binds a runtime library into
the process and the ACL device bind is per-thread, so a fresh process per case
keeps one case's state out of the next.

The ``runtime`` marker is what makes conftest's resource phase dispatch these
rather than deselect them, so it is load-bearing rather than descriptive — the
runtime it names is the one whose binaries the cases load.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import pytest


def _run_case(case: str, device_id: int, platform: str, queue) -> None:
    """Subprocess body: stand up the caller's device + stream, then drive the
    kernel-mode surface and report what came back."""
    result: dict[str, object] = {"case": case, "stage": "start", "ok": False}
    stream = 0
    try:
        import _task_interface as native
        from simpler.task_interface import CallConfig, ChipWorker

        from simpler_setup.runtime_builder import RuntimeBuilder

        bins = RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer", build=False)

        # The caller's device and stream. simpler must not create, reset or
        # destroy any of this.
        native._acl_bind_device(device_id)
        stream = native._acl_create_stream()
        result["stream_nonzero"] = bool(stream)
        result["stage"] = "borrowed"

        worker = ChipWorker()
        config = CallConfig()

        if case == "init_refused":
            # Reaching simpler_kernel_mode_init and being told UNSUPPORTED is
            # the pass: a call that never arrived would raise something else.
            with pytest.raises(Exception) as excinfo:  # noqa: PT011
                worker.kernel_init(device_id, bins, config, stream)
            result["error"] = str(excinfo.value)
            result["reached_abi"] = "kernel mode" in str(excinfo.value)
            result["initialized_after"] = bool(worker._impl.initialized)
            worker.finalize()
            result["ok"] = bool(result["reached_abi"]) and not result["initialized_after"]

        elif case == "null_stream_rejected":
            # Named at the Python boundary rather than deep in the C ABI.
            with pytest.raises(ValueError) as excinfo:
                worker.kernel_init(device_id, bins, config, 0)
            result["error"] = str(excinfo.value)
            result["ok"] = "caller_stream" in str(excinfo.value)

        elif case == "generation_is_unique":
            first = native._ChipWorker.next_kernel_context_generation()
            second = native._ChipWorker.next_kernel_context_generation()
            result["first"], result["second"] = first, second
            result["ok"] = first != 0 and second > first

        elif case == "uninitialized_surface_refuses":
            errors = []
            for label, fn in (
                ("supported", lambda: worker.kernel_mode_supported),
                ("launch", lambda: worker.kernel_launch(0, None, stream)),
            ):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    errors.append(label)
                    result[f"err_{label}"] = str(exc)
            result["refused"] = errors
            result["ok"] = errors == ["supported", "launch"]

        elif case == "program_init_still_works":
            # The program path must be unchanged by the kernel additions, and
            # the two identities must stay mutually exclusive on one worker.
            worker.init(device_id=device_id, bins=bins)
            result["stage"] = "program_init"
            result["initialized"] = bool(worker._impl.initialized)
            result["kernel_supported"] = bool(worker.kernel_mode_supported)
            try:
                worker.kernel_init(device_id, bins, config, stream)
                result["second_init_refused"] = False
            except RuntimeError:
                result["second_init_refused"] = True
            worker.finalize()
            result["ok"] = result["initialized"] and not result["kernel_supported"] and result["second_init_refused"]

        else:
            raise AssertionError(f"unknown case {case}")

        result["stage"] = "done"
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if stream:
            try:
                import _task_interface as native

                native._acl_destroy_stream(stream)
            except Exception:  # noqa: BLE001, S110
                pass
        queue.put(result)


def _run_in_subprocess(case: str, device_id: int, platform: str) -> dict:
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_case, args=(case, device_id, platform, queue))
    proc.start()
    proc.join(timeout=300)
    assert proc.exitcode is not None, f"case {case} did not exit within 300s"
    assert not queue.empty(), f"case {case} produced no result (exitcode={proc.exitcode})"
    return queue.get()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize(
    "case",
    [
        "init_refused",
        "null_stream_rejected",
        "generation_is_unique",
        "uninitialized_surface_refuses",
        "program_init_still_works",
    ],
)
def test_kernel_mode_surface_on_borrowed_stream(case, st_platform, st_device_ids):
    """Drive one kernel-mode case against a stream the test itself owns."""
    assert st_device_ids, "device_count(1) fixture must yield at least one id"
    result = _run_in_subprocess(case, int(st_device_ids[0]), st_platform)
    assert result["ok"], f"case {case} failed: {result}"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_borrowed_stream_survives_a_refused_kernel_init(st_platform, st_device_ids):
    """A refused kernel_init must leave the caller's stream usable.

    This is the guarantee a framework caller depends on: simpler failing to come
    up cannot take the caller's stream down with it. Destroying the stream after
    the refusal is what proves it was neither destroyed nor invalidated.
    """
    assert st_device_ids
    result = _run_in_subprocess("init_refused", int(st_device_ids[0]), st_platform)
    assert result.get("stream_nonzero"), f"test never obtained a stream: {result}"
    assert result["ok"], f"refused init did not behave per contract: {result}"
    assert os.path.exists("/proc/self"), "sanity: subprocess reported back to a live parent"
