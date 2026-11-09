# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real peer windows and independent data/epoch/progress checks for resident COUNTER waits."""

import json
import warnings
from functools import cache
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import (
    CallConfig,
    CommBufferSpec,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.hbg.native_causality import audit_native_protocol

HERE = Path(__file__).resolve().parent
MAX_ROWS = 256
WORDS = 32
ROW_WORDS = 64
SIGNAL_WORDS = 32
DIAG_WORDS = 16
DIAG_ROWS = 2 * MAX_ROWS + 4
GUARD = 0x13579BDF
POISON = -1000000007
GATE, READY_SEND, READY_WAIT, PUBLISHER = range(2 * MAX_ROWS, 2 * MAX_ROWS + 4)
VISITS, BODY_DONE, ERROR, EARLY, SEEN_SIGNAL, SEEN_TAG, ROWS, CAPACITY = range(8)
TIMEOUT, REGISTERED, CHECKSUM, PUBLISHED, ROLE, EPOCH, TAG, GUARD_FIELD = range(8, 16)
WINDOW_SHAPES = {
    "source": MAX_ROWS * ROW_WORDS,
    "payload": MAX_ROWS * ROW_WORDS,
    "signals": MAX_ROWS * SIGNAL_WORDS,
    "ready_payload": WORDS,
    "ready_signal": SIGNAL_WORDS,
}


def l3_case(function):
    function._st_level = 3
    return function


def source_values(rank, value_tag):
    result = torch.full((MAX_ROWS, ROW_WORDS), GUARD, dtype=torch.int32)
    result[:, :WORDS] = (
        value_tag * 100000
        + rank * 1000
        + torch.arange(MAX_ROWS, dtype=torch.int32).unsqueeze(1) * WORDS
        + torch.arange(WORDS, dtype=torch.int32)
        + 1
    )
    return result.flatten()


def empty_payload():
    result = torch.full((MAX_ROWS, ROW_WORDS), GUARD, dtype=torch.int32)
    result[:, :WORDS] = POISON
    return result.flatten()


def empty_signals(rows):
    result = torch.full((rows, SIGNAL_WORDS), GUARD, dtype=torch.int32)
    result[:, 0] = 0
    return result.flatten()


def empty_ready_payload():
    result = torch.full((WORDS,), GUARD, dtype=torch.int32)
    result[:8] = 0
    return result


def stage(buffer, tensor):
    torch.frombuffer(buffer.shm.buf, dtype=tensor.dtype, count=tensor.numel()).copy_(tensor.flatten())


def read(buffer, count):
    return torch.frombuffer(buffer.shm.buf, dtype=torch.int32, count=count).clone()


def close_after_failure(worker):
    try:
        worker.close()
    except Exception as error:
        warnings.warn(f"peer COUNTER cleanup also failed: {error}", RuntimeWarning, stacklevel=2)


@cache
def compile_peer_counter(platform):
    specifications = (
        (0, "aiv/waiter.cpp", "aiv", 4),
        (1, "aic/registration_gate.cpp", "aic", 2),
        (2, "aiv/ready_send.cpp", "aiv", 4),
        (3, "aiv/publish.cpp", "aiv", 5),
        (4, "aiv/consume.cpp", "aiv", 4),
    )
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/peer_counter.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 8 + [D.SCALAR] * 6,
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels" / source),
                "core_type": core,
                "signature": [D.INOUT] * tensors,
            }
            for index, source, core, tensors in specifications
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), platform, "peer_counter", _pto_isa_compile_cache_token())
    )


class PeerCounterRunner:
    """Keep both peer allocations alive until their enclosing L3 invocation completes."""

    def __init__(self, platform, devices, directory, *, pressure=False, diagnostics=4):
        assert len(devices) >= 2, "peer COUNTER qualification needs two assigned A5 devices"
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pressure = pressure
        self.diagnostics = diagnostics
        assert diagnostics in (0, 4)
        self.iteration = 0
        self.prior_run_epochs = [None, None]
        self.diag = [torch.zeros((DIAG_ROWS, DIAG_WORDS), dtype=torch.int64).share_memory_() for _ in range(2)]
        self.output = [empty_payload().share_memory_() for _ in range(2)]
        self.registration = [torch.full((DIAG_WORDS,), GUARD, dtype=torch.int64).share_memory_() for _ in range(2)]
        self.worker = Worker(
            level=3,
            platform=platform,
            runtime="host_build_graph",
            device_ids=list(map(int, devices[:2])),
            num_sub_workers=0,
        )
        try:
            self.callable_handle = self.worker.register(compile_peer_counter(platform))
            self.worker.init()
            self.staging = [
                {name: self.worker.create_buffer(count * 4) for name, count in WINDOW_SHAPES.items()} for _ in range(2)
            ]
        except BaseException:
            close_after_failure(self.worker)
            raise

    def _args(self, domain, role, epoch, value_tag):
        args = TaskArgs()
        for name, count in WINDOW_SHAPES.items():
            args.add_tensor(domain[role].buffers[name].tensor((count,), DataType.INT32), TensorArgType.INOUT)
        for tensor, dtype in (
            (self.diag[role], DataType.INT64),
            (self.output[role], DataType.INT32),
            (self.registration[role], DataType.INT64),
        ):
            args.add_tensor(self.worker.make_tensor_arg(tensor, (tensor.numel(),), dtype), TensorArgType.INOUT)
        for value in (domain.device_context(role), role, epoch, value_tag, int(self.pressure), MAX_ROWS):
            args.add_scalar(int(value))
        return args

    def _save_diagnostics(self, capture, epoch, value_tag, reset, error=None):
        evidence = {
            "signal_epoch": epoch,
            "value_tag": value_tag,
            "reset_window": reset,
            "pressure": self.pressure,
            "error": error,
            "host_diagnostics": [value.tolist() for value in self.diag],
            "host_registration": [value.tolist() for value in self.registration],
            "host_output": [value.tolist() for value in self.output],
            "captures": [str(path.relative_to(capture)) for path in capture.rglob("chip_swimlane_records.json")],
            "note": ("On a failed invocation these Host buffers do not prove that device copyback completed."),
        }
        (capture / "peer-counter-diagnostics.json").write_text(json.dumps(evidence, indent=2) + "\n")

    def run(self, epoch, value_tag, *, reset):
        for rank in range(2):
            self.diag[rank].zero_()
            self.diag[rank][:, GUARD_FIELD] = GUARD
            self.output[rank].copy_(empty_payload())
            self.registration[rank].fill_(GUARD)
            self.registration[rank][0] = 0
        self.iteration += 1
        capture = self.directory / f"round_{self.iteration}_signal_{epoch}_tag_{value_tag}"
        capture.mkdir()
        config = CallConfig()
        config.enable_chip_swimlane = self.diagnostics
        config.output_prefix = str(capture)
        inputs = [source_values(rank, value_tag) for rank in range(2)]

        def orchestrate(orch, _args, _config):
            buffers = [
                CommBufferSpec(name=name, dtype="int32", count=count, nbytes=count * 4)
                for name, count in WINDOW_SHAPES.items()
            ]
            domain = orch.get_or_allocate_domain(
                name="resident_peer_counter",
                members=[0, 1],
                window_size=sum(item.nbytes for item in buffers),
                buffers=buffers,
                retain_after_run=True,
            )
            for rank in range(2):
                stage(self.staging[rank]["source"], inputs[rank])
                orch.copy_to(domain[rank].buffers["source"], self.staging[rank]["source"])
                if reset:
                    for name, initial in (
                        ("payload", empty_payload()),
                        ("signals", empty_signals(MAX_ROWS)),
                        ("ready_payload", empty_ready_payload()),
                        ("ready_signal", empty_signals(1)),
                    ):
                        stage(self.staging[rank][name], initial)
                        orch.copy_to(domain[rank].buffers[name], self.staging[rank][name])
            for rank in range(2):
                rank_config = CallConfig()
                rank_config.enable_chip_swimlane = self.diagnostics
                rank_config.output_prefix = str(capture / f"rank_{rank}")
                Path(rank_config.output_prefix).mkdir()
                orch.submit_next_level(
                    self.callable_handle, self._args(domain, rank, epoch, value_tag), rank_config, worker=rank
                )

        failure = None
        try:
            self.worker.submit(orchestrate, config=config).wait(timeout=30)
            self._check(capture, epoch, value_tag, inputs)
        except BaseException as error:
            failure = repr(error)
            raise
        finally:
            self._save_diagnostics(capture, epoch, value_tag, reset, failure)

    def _audit_resident_capture(self, capture, rank):
        paths = list((capture / f"rank_{rank}").rglob("chip_swimlane_records.json"))
        assert len(paths) == 1, paths
        raw = json.loads(paths[0].read_text())
        protocol = raw.get("aicore_protocol", {})
        assert protocol.get("producer") == "aicore", paths[0]
        assert protocol.get("schema") in ("hbg-native-protocol-v8", "hbg-native-protocol-v9"), (
            "resident diagnostic qualification requires resource-scoped native evidence",
            paths[0],
        )
        identities = raw["aicore_execution_identities"]["records"]
        audit = audit_native_protocol(protocol, identities)
        assert audit["protocol_verified"]
        epoch = protocol["run_epoch"]
        prior = self.prior_run_epochs[rank]
        assert prior is None or epoch > prior, (rank, prior, epoch)
        self.prior_run_epochs[rank] = epoch
        return protocol, audit

    def _check_business_progress(self, count, capacity, epoch, value_tag):
        expected_rows = ({READY_WAIT, PUBLISHER}, {*range(count), *range(MAX_ROWS, MAX_ROWS + count), GATE, READY_SEND})
        for rank, rows in enumerate(expected_rows):
            diag = self.diag[rank]
            assert torch.all(diag[:, GUARD_FIELD] == GUARD), (rank, "diagnostic guard")
            assert not torch.count_nonzero(diag[:, [ERROR, EARLY, TIMEOUT]]), (rank, diag.tolist())
            expected = torch.zeros(DIAG_ROWS, dtype=torch.int64)
            expected[list(rows)] = 1
            torch.testing.assert_close(diag[:, VISITS], expected, rtol=0, atol=0)
            torch.testing.assert_close(diag[:, BODY_DONE], expected, rtol=0, atol=0)
            assert torch.all(diag[list(rows), EPOCH] == epoch)
            assert torch.all(diag[list(rows), TAG] == value_tag)
            assert torch.all(self.registration[rank][1:] == GUARD)
        assert int(self.registration[0][0]) == 1
        assert int(self.registration[1][0]) == count
        gate = self.diag[1][GATE]
        assert int(gate[ROWS]) == int(gate[REGISTERED]) == count
        assert int(gate[CAPACITY]) == capacity
        assert int(self.diag[0][PUBLISHER][PUBLISHED]) == count
        assert int(self.diag[1][READY_SEND][PUBLISHED]) == 1
        assert torch.all(self.diag[1][:count, SEEN_SIGNAL] < epoch), "main notifications arrived before registration"

    def _copy_windows(self):
        domain = self.worker.live_domains["resident_peer_counter"]
        values = []
        for rank in range(2):
            current = {}
            for name, count in WINDOW_SHAPES.items():
                self.worker.copy_from(self.staging[rank][name], domain[rank].buffers[name])
                current[name] = read(self.staging[rank][name], count)
            values.append(current)
        return values

    def _check_business_behavior(self, capture, epoch, value_tag, inputs):
        # The gate records the public rt_available_aiv_count() value. Native
        # trace topology is cross-checked separately as diagnostic qualification.
        capacity = int(self.diag[1][GATE][CAPACITY])
        assert 0 < capacity <= MAX_ROWS
        count = capacity + 1 if self.pressure else 1
        assert count <= MAX_ROWS
        self._check_business_progress(count, capacity, epoch, value_tag)
        windows = self._copy_windows()
        (capture / "peer-counter-windows.json").write_text(
            json.dumps([{name: tensor.tolist() for name, tensor in rank.items()} for rank in windows], indent=2) + "\n"
        )
        expected_payload = empty_payload().reshape(MAX_ROWS, ROW_WORDS)
        expected_payload[:count, :WORDS] = inputs[0].reshape(MAX_ROWS, ROW_WORDS)[:count, :WORDS]
        expected_output = empty_payload().reshape(MAX_ROWS, ROW_WORDS)
        expected_output[:count, :WORDS] = expected_payload[:count, :WORDS] + 7
        expected_signals = empty_signals(MAX_ROWS).reshape(MAX_ROWS, SIGNAL_WORDS)
        expected_signals[:count, 0] = epoch
        ready_packet = torch.full((WORDS,), GUARD, dtype=torch.int32)
        ready_packet[:8] = torch.tensor([1, epoch, value_tag, count, count, 1, 0, capacity], dtype=torch.int32)
        ready_signal = empty_signals(1)
        ready_signal[0] = epoch
        for rank in range(2):
            torch.testing.assert_close(windows[rank]["source"], inputs[rank], rtol=0, atol=0)
            torch.testing.assert_close(windows[rank]["ready_payload"], ready_packet, rtol=0, atol=0)
        torch.testing.assert_close(windows[0]["payload"], empty_payload(), rtol=0, atol=0)
        torch.testing.assert_close(windows[1]["payload"], expected_payload.flatten(), rtol=0, atol=0)
        torch.testing.assert_close(windows[0]["signals"], empty_signals(MAX_ROWS), rtol=0, atol=0)
        torch.testing.assert_close(windows[1]["signals"], expected_signals.flatten(), rtol=0, atol=0)
        torch.testing.assert_close(windows[0]["ready_signal"], ready_signal, rtol=0, atol=0)
        torch.testing.assert_close(windows[1]["ready_signal"], empty_signals(1), rtol=0, atol=0)
        torch.testing.assert_close(self.output[0], empty_payload(), rtol=0, atol=0)
        torch.testing.assert_close(self.output[1], expected_output.flatten(), rtol=0, atol=0)
        return count, capacity

    def _check_resident_qualification(self, capture, count, capacity):
        receiver, receiver_audit = self._audit_resident_capture(capture, 1)
        source, source_audit = self._audit_resident_capture(capture, 0)
        assert sum(worker["core_type"] == 1 for worker in receiver["workers"]) == capacity, (
            "native worker topology differs from the public AIV capacity"
        )
        # Payload, epoch and registration gates are checked by the business
        # oracle. The native audit verifies execution and dependency completion.
        assert count > 0
        (capture / "peer-counter-audit.json").write_text(
            json.dumps(
                {"capacity_aiv": capacity, "main_waiters": count, "source": source_audit, "receiver": receiver_audit},
                indent=2,
            )
            + "\n"
        )

    def _check(self, capture, epoch, value_tag, inputs):
        try:
            count, capacity = self._check_business_behavior(capture, epoch, value_tag, inputs)
        except AssertionError as error:
            raise AssertionError(f"peer COUNTER behavior check failed: {error}") from error
        if not self.diagnostics:
            return
        try:
            self._check_resident_qualification(capture, count, capacity)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError(f"A5 resident diagnostic qualification failed: {error}") from error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self.worker.close()
        else:
            close_after_failure(self.worker)
