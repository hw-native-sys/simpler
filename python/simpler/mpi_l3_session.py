# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI L3 group runner driven by one rank-0 named shared-memory mailbox."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .global_comm_domain import (
    GlobalDomainCommand,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    decode_descriptor_table,
    encode_descriptor_table,
    validate_descriptor_table,
)
from .mpi_group_mailbox import (
    MailboxGroupState,
    MailboxOpcode,
    MailboxRequest,
    MailboxRequestState,
    MailboxTarget,
    MpiGroupMailbox,
    MpiRankError,
    open_rank_mailbox,
)
from .remote_l3_protocol import (
    FrameHeader,
    FrameType,
    decode_frame,
    encode_frame,
)
from .remote_l3_session import (
    _format_remote_error,
    _install_manifest_dispatcher_registry,
    _install_manifest_inner_registry,
    _run_command_loop,
    _startup_deadline,
)
from .worker import Worker

_POLL_WAIT = threading.Event()


def _cooperative_poll_backoff(spins: int) -> int:
    """Bound a hot poll with a scheduler yield, without sleep-based polling."""
    spins += 1
    if spins < 64:
        return spins
    if hasattr(os, "sched_yield"):
        os.sched_yield()
    else:
        _POLL_WAIT.wait(0.0002)
    return 0


class MpiGlobalDomainExchange:
    """Run Global CommDomain collectives on the dispatcher thread."""

    def __init__(self, comm: Any, *, group_worker_ids: tuple[int, ...], timeout_s: float) -> None:
        self._comm = comm
        self._rank = int(comm.Get_rank())
        self._group_worker_ids = tuple(int(worker_id) for worker_id in group_worker_ids)
        self._group_worker_id_set = set(self._group_worker_ids)
        self._timeout_s = float(timeout_s)
        if not (self._timeout_s > 0 and math.isfinite(self._timeout_s)):
            raise ValueError("MPI Global CommDomain timeout must be a positive finite number")

    def _allgather(self, payload: Any, *, operation: str, on_timeout) -> list[Any]:
        # The mpi4py version used by the deployment has no non-blocking pickle
        # convenience named ``iallgather``. ``Iallgather`` is the buffer API,
        # while lowercase ``allgather`` pickles Python objects and returns a
        # list. The outer L4 mailbox timeout remains the group-level watchdog
        # and terminates the complete mpirun process group if this call stalls.
        return list(self._comm.allgather(payload))

    def prepare_import(self, command: GlobalDomainCommand, inner_worker: Worker, worker_id: int) -> bytes | None:
        if command.phase is not GlobalDomainPhase.PREPARE_EXPORT:
            return None
        if {int(member.node_worker_id) for member in command.members} != self._group_worker_id_set:
            return None

        ok = True
        error_message = ""
        local_payload = b""
        release_command = GlobalDomainReleaseCommand(command.domain_id, command.generation)

        def release_local() -> None:
            inner_worker._release_global_domain_node(  # noqa: SLF001
                release_command,
                suppress_errors=True,
            )

        try:
            descriptors = inner_worker._prepare_global_domain_node(command, int(worker_id))  # noqa: SLF001
            local_payload = encode_descriptor_table(descriptors)
        except BaseException as exc:  # noqa: BLE001
            release_local()
            ok = False
            error_message = _format_remote_error(
                f"mpi global domain prepare rank={self._rank} worker_id={worker_id}",
                exc,
            )

        try:
            gathered = self._allgather(
                (self._rank, ok, error_message, local_payload),
                operation="prepare",
                on_timeout=release_local,
            )
        except BaseException:
            release_local()
            raise
        errors = [(rank, message) for rank, rank_ok, message, _payload in gathered if not rank_ok]
        if errors:
            if ok:
                release_local()
            rank, message = errors[0]
            raise RuntimeError(f"MPI Global CommDomain prepare failed on rank {rank}: {message}")

        try:
            descriptor_by_rank = {}
            for _rank, _ok, _message, payload in sorted(gathered, key=lambda item: int(item[0])):
                for descriptor in decode_descriptor_table(bytes(payload)):
                    if descriptor.domain_rank in descriptor_by_rank:
                        raise RuntimeError("MPI Global CommDomain exchange returned a duplicate domain rank")
                    descriptor_by_rank[descriptor.domain_rank] = descriptor
            descriptors = tuple(descriptor_by_rank[rank] for rank in range(len(command.members)))
            validate_descriptor_table(descriptors, rank_count=len(command.members), profile=command.profile)
        except BaseException:
            release_local()
            raise

        import_command = GlobalDomainCommand(
            phase=GlobalDomainPhase.IMPORT,
            domain_id=command.domain_id,
            generation=command.generation,
            name=command.name,
            profile=command.profile,
            window_size=command.window_size,
            members=command.members,
            buffers=command.buffers,
            descriptors=descriptors,
        )
        import_ok = True
        import_error = ""
        try:
            inner_worker._import_global_domain_node(import_command, int(worker_id))  # noqa: SLF001
        except BaseException as exc:  # noqa: BLE001
            release_local()
            import_ok = False
            import_error = _format_remote_error(
                f"mpi global domain import rank={self._rank} worker_id={worker_id}",
                exc,
            )

        try:
            statuses = self._allgather(
                (self._rank, import_ok, import_error),
                operation="import",
                on_timeout=release_local,
            )
        except BaseException:
            release_local()
            raise
        import_errors = [(rank, message) for rank, rank_ok, message in statuses if not rank_ok]
        if import_errors:
            release_local()
            rank, message = import_errors[0]
            raise RuntimeError(f"MPI Global CommDomain import failed on rank {rank}: {message}")
        return encode_descriptor_table(descriptors)


@dataclass
class _DomainRequest:
    exchange: MpiGlobalDomainExchange
    command: GlobalDomainCommand
    inner_worker: Worker
    worker_id: int
    done: threading.Event
    result: bytes | None = None
    error: BaseException | None = None

    def execute(self) -> None:
        try:
            self.result = self.exchange.prepare_import(self.command, self.inner_worker, self.worker_id)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self.done.set()


class _DomainBridge:
    def __init__(self, events: queue.Queue[tuple[str, Any]], exchange: MpiGlobalDomainExchange) -> None:
        self._events = events
        self._exchange = exchange

    def prepare_import(self, command: GlobalDomainCommand, inner_worker: Worker, worker_id: int) -> bytes | None:
        request = _DomainRequest(
            self._exchange,
            command,
            inner_worker,
            int(worker_id),
            threading.Event(),
        )
        self._events.put(("domain", request))
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result


class _InMemoryCommandConnection:
    """Socket-shaped blocking stream backed by queues, with no OS socket."""

    def __init__(self) -> None:
        self._incoming: queue.Queue[bytes] = queue.Queue()
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._recv_buffer = bytearray()

    @property
    def events(self) -> queue.Queue[tuple[str, Any]]:
        return self._events

    def recv(self, nbytes: int) -> bytes:
        while not self._recv_buffer:
            self._recv_buffer.extend(self._incoming.get())
        count = min(int(nbytes), len(self._recv_buffer))
        data = bytes(self._recv_buffer[:count])
        del self._recv_buffer[:count]
        return data

    def sendall(self, data: bytes) -> None:
        self._events.put(("reply", bytes(data)))

    def feed(self, frame: bytes) -> None:
        self._incoming.put(bytes(frame))

    def next_reply(self) -> bytes:
        while True:
            kind, value = self._events.get()
            if kind == "domain":
                value.execute()
                continue
            if kind == "processor_error":
                error_type, message = value
                raise RuntimeError(f"{error_type}: {message}")
            if kind != "reply":
                raise RuntimeError(f"MPI command processor emitted unknown event {kind!r}")
            return bytes(value)

    def exchange(self, frame: bytes) -> bytes:
        self.feed(frame)
        return self.next_reply()


def _load_group_manifest_from_rank0(comm: Any, manifest_path: str) -> dict[str, Any]:
    rank = int(comm.Get_rank())
    if rank == 0:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                payload = (True, json.load(f))
        except BaseException as exc:  # noqa: BLE001
            payload = (False, f"{type(exc).__name__}: {exc}")
    else:
        payload = None
    ok, value = comm.bcast(payload, root=0)
    if not ok:
        raise RuntimeError(f"MPI L3 rank0 failed to read group manifest {manifest_path!r}: {value}")
    if not isinstance(value, dict):
        raise ValueError("MPI L3 group manifest broadcast returned a non-object payload")
    return value


def _rewrite_frame_identity(
    frame_bytes: bytes,
    manifest: dict[str, Any],
    *,
    sequence: int | None = None,
) -> bytes:
    frame = decode_frame(frame_bytes)
    header = FrameHeader(
        frame_type=frame.header.frame_type,
        session_id=int(manifest["session_id"]),
        worker_id=int(manifest["worker_id"]),
        sequence=frame.header.sequence if sequence is None else int(sequence),
        flags=frame.header.flags,
    )
    return encode_frame(header, frame.payload)


def _reply_error(reply_bytes: bytes) -> tuple[str, str] | None:
    frame = decode_frame(reply_bytes)
    if frame.header.frame_type is FrameType.COMPLETION:
        if len(frame.payload) < 16:
            return ("ProtocolError", "truncated completion reply")
        _sequence, error_code, message_bytes = struct.unpack_from("<QiI", frame.payload)
        offset = 16
    elif frame.header.frame_type is FrameType.CONTROL_REPLY:
        if len(frame.payload) < 24:
            return ("ProtocolError", "truncated control reply")
        _sequence, _control, _version, error_code, message_bytes = struct.unpack_from("<QIIiI", frame.payload)
        offset = 24
    else:
        return None
    if message_bytes > len(frame.payload) - offset:
        return ("ProtocolError", "reply error message is truncated")
    if error_code == 0:
        return None
    message = frame.payload[offset : offset + message_bytes].decode("utf-8", errors="replace")
    return ("RemoteOperationError", message)


def _payload_for_rank(request: MailboxRequest, rank: int) -> bytes | None:
    if request.target is MailboxTarget.GROUP:
        return request.payloads[0]
    if request.target is MailboxTarget.RANK:
        return request.payloads[0] if rank == request.target_rank else None
    if request.target is MailboxTarget.PER_RANK:
        return request.payloads[rank]
    raise ValueError(f"unsupported MPI mailbox target {request.target}")


def _shutdown_frame(manifest: dict[str, Any]) -> bytes:
    return encode_frame(
        FrameHeader(
            frame_type=FrameType.SHUTDOWN,
            session_id=int(manifest["session_id"]),
            worker_id=int(manifest["worker_id"]),
            sequence=0,
        ),
        b"",
    )


def _select_mailbox_replies(
    request: MailboxRequest,
    gathered: list[tuple[int, bool, bytes, MpiRankError | None]],
    worker_ids: tuple[int, ...],
) -> tuple[bytes, ...]:
    def _for_request(reply: bytes, request_frame: bytes) -> bytes:
        if not reply:
            return b""
        decoded_reply = decode_frame(reply)
        decoded_request = decode_frame(request_frame)
        reply_payload = bytearray(decoded_reply.payload)
        if decoded_reply.header.frame_type in (FrameType.COMPLETION, FrameType.CONTROL_REPLY):
            if len(reply_payload) < 8:
                raise RuntimeError("MPI rank reply is truncated before its sequence field")
            struct.pack_into("<Q", reply_payload, 0, decoded_request.header.sequence)
        return encode_frame(
            FrameHeader(
                frame_type=decoded_reply.header.frame_type,
                session_id=decoded_request.header.session_id,
                worker_id=decoded_request.header.worker_id,
                sequence=decoded_request.header.sequence,
                flags=decoded_reply.header.flags,
            ),
            bytes(reply_payload),
        )

    replies_by_rank = {int(result_rank): bytes(reply) for result_rank, executed, reply, _error in gathered if executed}
    if request.target is MailboxTarget.GROUP and request.payloads and request.payloads[0]:
        requested_worker_id = int(decode_frame(request.payloads[0]).header.worker_id)
        if requested_worker_id in worker_ids:
            requested_rank = worker_ids.index(requested_worker_id)
            return (_for_request(replies_by_rank.get(requested_rank, b""), request.payloads[0]),)
    if request.target is MailboxTarget.RANK:
        return (_for_request(replies_by_rank.get(request.target_rank, b""), request.payloads[0]),)
    if request.target is MailboxTarget.PER_RANK:
        return tuple(
            _for_request(replies_by_rank.get(reply_rank, b""), request.payloads[reply_rank])
            for reply_rank in range(len(request.payloads))
        )
    return tuple(replies_by_rank[reply_rank] for reply_rank in sorted(replies_by_rank)) or (b"",)


def _run_group_session(  # noqa: PLR0912, PLR0915 -- startup, ordered dispatch, gather, and teardown stay linear
    *,
    dispatch_comm: Any,
    domain_comm: Any,
    group_manifest: dict[str, Any],
    manifest: dict[str, Any],
) -> int:
    rank = int(dispatch_comm.Get_rank())
    worker_ids = tuple(int(value) for value in group_manifest["worker_ids"])
    mailbox: MpiGroupMailbox | None = None
    inner_worker = Worker(
        level=3,
        platform=str(manifest["platform"]),
        runtime=str(manifest.get("runtime", "tensormap_and_ringbuffer")),
        device_ids=tuple(int(value) for value in manifest.get("device_ids", ())),
        num_sub_workers=int(manifest.get("num_sub_workers", 0)),
        heap_ring_size=int(manifest["heap_ring_size"]) if manifest.get("heap_ring_size") is not None else None,
    )
    connection = _InMemoryCommandConnection()
    processor_thread: threading.Thread | None = None
    startup_ok = True
    startup_error = ""

    try:
        dispatch_registry = _install_manifest_dispatcher_registry(manifest)
        inner_handles = _install_manifest_inner_registry(manifest, inner_worker)
        inner_worker.init(_startup_deadline=_startup_deadline(manifest))
        exchange = MpiGlobalDomainExchange(
            domain_comm,
            group_worker_ids=worker_ids,
            timeout_s=float(manifest["session_timeout_s"]),
        )
        bridge = _DomainBridge(connection.events, exchange)

        def _processor() -> None:
            try:
                _run_command_loop(
                    connection,  # type: ignore[arg-type]
                    manifest,
                    inner_worker,
                    inner_handles,
                    dispatch_registry,
                    bridge.prepare_import,
                )
            except BaseException as exc:  # noqa: BLE001
                connection.events.put(("processor_error", (type(exc).__name__, str(exc))))

        processor_thread = threading.Thread(target=_processor, name=f"simpler-mpi-command-rank-{rank}")
        processor_thread.start()
        hello = decode_frame(connection.next_reply())
        if hello.header.frame_type is not FrameType.HELLO:
            raise RuntimeError("MPI command processor did not publish HELLO")
    except BaseException as exc:  # noqa: BLE001
        startup_ok = False
        startup_error = _format_remote_error(f"MPI rank {rank} startup", exc)

    readiness = dispatch_comm.allgather((rank, startup_ok, startup_error))
    mailbox_ready = True
    mailbox_error = ""
    if rank == 0:
        try:
            mailbox = open_rank_mailbox(group_manifest["mailbox"], rank=rank)
            assert mailbox is not None
            failures = [(ready_rank, error) for ready_rank, ok, error in readiness if not ok]
            if failures:
                mailbox_error = "; ".join(f"rank {ready_rank}: {error}" for ready_rank, error in failures)
                mailbox.mark_terminal(mailbox_error)
                mailbox_ready = False
            else:
                mailbox.publish_ready()
        except BaseException as exc:  # noqa: BLE001
            mailbox_ready = False
            mailbox_error = _format_remote_error("MPI rank 0 mailbox startup", exc)
            if mailbox is not None:
                mailbox.mark_terminal(mailbox_error)
    mailbox_ready, mailbox_error = dispatch_comm.bcast((mailbox_ready, mailbox_error), root=0)
    if not mailbox_ready:
        if processor_thread is not None:
            connection.feed(_shutdown_frame(manifest))
            processor_thread.join(timeout=1.0)
        inner_worker.close()
        if mailbox is not None:
            mailbox.close()
        return 1

    last_sequence_id = 0
    local_command_sequence = 0
    session_timeout_s = float(manifest["session_timeout_s"])
    exit_code = 0
    try:
        while True:
            if rank == 0:
                assert mailbox is not None
                request_wait_deadline = time.monotonic() + session_timeout_s
                spins = 0
                while True:
                    request_state = mailbox.request_state
                    if request_state in (
                        MailboxRequestState.REQUEST_READY,
                        MailboxRequestState.SHUTDOWN_READY,
                    ):
                        break
                    if mailbox.group_state is MailboxGroupState.TERMINAL:
                        break
                    now = time.monotonic()
                    if request_state is MailboxRequestState.IDLE:
                        request_wait_deadline = now + session_timeout_s
                    elif now >= request_wait_deadline:
                        mailbox.mark_terminal(f"MPI group mailbox request lane stalled in state {request_state.name}")
                        break
                    spins = _cooperative_poll_backoff(spins)
                if mailbox.group_state is MailboxGroupState.TERMINAL:
                    request = None
                else:
                    try:
                        request = mailbox.accept_request(last_sequence_id=last_sequence_id)
                        last_sequence_id = request.sequence_id
                    except Exception as exc:
                        mailbox.mark_terminal(_format_remote_error("MPI rank 0 mailbox request", exc))
                        request = None
            else:
                request = None
            request = dispatch_comm.bcast(request, root=0)
            if request is None:
                exit_code = 1
                break

            local_reply = b""
            local_error: MpiRankError | None = None
            payload = None
            try:
                payload = _payload_for_rank(request, rank)
                if request.opcode is MailboxOpcode.PING:
                    local_reply = b""
                elif request.opcode is MailboxOpcode.SHUTDOWN:
                    assert payload is not None
                    connection.feed(_rewrite_frame_identity(payload, manifest))
                elif payload is not None:
                    local_command_sequence += 1
                    local_reply = connection.exchange(
                        _rewrite_frame_identity(payload, manifest, sequence=local_command_sequence)
                    )
            except BaseException as exc:  # noqa: BLE001
                local_error = MpiRankError(rank, type(exc).__name__, str(exc))

            gathered = dispatch_comm.gather((rank, payload is not None, local_reply, local_error), root=0)
            terminal_dispatch_failure = False
            if rank == 0:
                assert mailbox is not None and gathered is not None
                transport_errors = tuple(item[3] for item in gathered if item[3] is not None)
                application_errors: list[MpiRankError] = []
                for result_rank, executed, reply, _error in gathered:
                    if not executed or not reply:
                        continue
                    decoded_error = _reply_error(reply)
                    if decoded_error is not None:
                        error_type, message = decoded_error
                        application_errors.append(MpiRankError(int(result_rank), error_type, message))
                if transport_errors:
                    mailbox.fail_request(
                        sequence_id=request.sequence_id,
                        errors=transport_errors,
                        terminal=True,
                    )
                    exit_code = 1
                    terminal_dispatch_failure = True
                elif application_errors:
                    mailbox.fail_request(
                        sequence_id=request.sequence_id,
                        errors=tuple(application_errors),
                        terminal=False,
                    )
                elif request.opcode is MailboxOpcode.SHUTDOWN:
                    mailbox.complete_shutdown(sequence_id=request.sequence_id)
                    mailbox.publish_closed()
                else:
                    replies = _select_mailbox_replies(request, gathered, worker_ids)
                    mailbox.complete_request(sequence_id=request.sequence_id, payloads=replies)
            terminal_dispatch_failure = dispatch_comm.bcast(terminal_dispatch_failure, root=0)
            if terminal_dispatch_failure:
                break
            if request.opcode is MailboxOpcode.SHUTDOWN:
                break
    finally:
        if processor_thread is not None:
            if processor_thread.is_alive():
                connection.feed(_shutdown_frame(manifest))
            processor_thread.join(timeout=5.0)
        try:
            inner_worker.close()
        finally:
            if mailbox is not None:
                mailbox.close()
    return exit_code


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", required=True)
    ns = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("simpler.mpi_l3_session requires mpi4py inside the mpirun Python environment") from exc

    world = MPI.COMM_WORLD
    group_manifest = _load_group_manifest_from_rank0(world, ns.group_manifest)
    rank_manifests = group_manifest.get("rank_manifests")
    if not isinstance(rank_manifests, list):
        raise ValueError("MPI L3 group manifest requires a rank_manifests list")
    world_size = int(world.Get_size())
    rank = int(world.Get_rank())
    if len(rank_manifests) != world_size or int(group_manifest.get("world_size", -1)) != world_size:
        raise ValueError("MPI L3 group manifest world size does not match MPI_COMM_WORLD")
    if len(group_manifest.get("worker_ids", ())) != world_size:
        raise ValueError("MPI L3 group manifest worker_ids must match MPI_COMM_WORLD size")

    dispatch_comm = world.Dup()
    domain_comm = world.Dup()
    try:
        return _run_group_session(
            dispatch_comm=dispatch_comm,
            domain_comm=domain_comm,
            group_manifest=group_manifest,
            manifest=dict(rank_manifests[rank]),
        )
    finally:
        domain_comm.Free()
        dispatch_comm.Free()


if __name__ == "__main__":
    sys.exit(main())
