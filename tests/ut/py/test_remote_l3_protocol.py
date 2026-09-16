# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import struct

import pytest
from simpler import remote_l3_session
from simpler.remote_l3_protocol import (
    MAX_ERROR_BYTES,
    REMOTE_BUFFER_ACCESS_READ,
    CallableKind,
    ControlName,
    ExportBufferResult,
    ImportBufferResult,
    RemoteAddressSpace,
    RemoteRegistryTarget,
    decode_export_buffer_result,
    decode_register_callable_command,
    decode_task_payload,
    encode_completion,
    encode_control_reply,
    encode_export_buffer_result,
    encode_import_buffer_result,
    encode_register_callable_command,
)

from simpler_setup.tools import containment


def _oversized_multibyte_error_message():
    multi_byte = chr(0x20AC)
    return multi_byte * (MAX_ERROR_BYTES // len(multi_byte.encode("utf-8")) + 1)


def test_task_payload_decode_preserves_scope_stats_config():
    prefix = b"/tmp/remote-scope"
    config = struct.pack("<iiiiii", 5, 0, 0, 0, 0, 1) + struct.pack("<I", len(prefix)) + prefix
    args = struct.pack("<III", 0, 0, 0)
    wire = (b"\xab" * 32) + config + args

    payload = decode_task_payload(wire)

    assert payload.config.aicpu_thread_num == 5
    assert payload.config.enable_scope_stats is True
    assert payload.config.output_prefix == prefix.decode()


def test_register_callable_command_round_trips_python_import_target():
    digest = b"\x12" * 32
    target = b"pkg.mod:remote_entry"
    encoded = encode_register_callable_command(
        RemoteRegistryTarget.REMOTE_TASK_DISPATCHER,
        CallableKind.PYTHON_IMPORT,
        digest,
        1,
        target,
    )

    decoded = decode_register_callable_command(encoded)

    assert decoded.target_registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER
    assert decoded.callable_kind == CallableKind.PYTHON_IMPORT
    assert decoded.digest == digest
    assert decoded.payload_version == 1
    assert decoded.payload == target


def test_completion_rejects_oversized_utf8_error_message():
    message = _oversized_multibyte_error_message()

    with pytest.raises(ValueError, match="completion error message too long"):
        encode_completion(1, 1, message)


def test_control_reply_rejects_oversized_utf8_error_message():
    message = _oversized_multibyte_error_message()

    with pytest.raises(ValueError, match="control reply error message too long"):
        encode_control_reply(1, ControlName.PREPARE_CALLABLE, 1, 1, message)


def _export_result(**overrides):
    fields = {
        "owner_worker_id": 1,
        "buffer_id": 2,
        "generation": 3,
        "address_space": RemoteAddressSpace.REMOTE_WINDOW,
        "offset": 0,
        "nbytes": 4,
        "export_id": 5,
        "remote_addr": 0,
        "rkey_or_token": 0,
        "ub_ldst_va": 0,
        "access_flags": REMOTE_BUFFER_ACCESS_READ,
        "transport_profile": "sim",
        "transport_descriptor": b"",
    }
    fields.update(overrides)
    return ExportBufferResult(**fields)


def _import_result(**overrides):
    fields = {
        "importer_worker_id": 1,
        "owner_worker_id": 2,
        "buffer_id": 3,
        "generation": 4,
        "import_id": 5,
        "address_space": RemoteAddressSpace.REMOTE_WINDOW,
        "offset": 0,
        "nbytes": 4,
        "remote_addr": 0,
        "rkey_or_token": 0,
        "ub_ldst_va": 0,
        "access_flags": REMOTE_BUFFER_ACCESS_READ,
        "transport_profile": "sim",
        "import_descriptor": b"",
    }
    fields.update(overrides)
    return ImportBufferResult(**fields)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_worker_id", -1),
        ("buffer_id", 0),
        ("generation", 0),
    ],
)
def test_export_buffer_result_rejects_invalid_live_identity(field, value):
    with pytest.raises(ValueError, match="live owner buffer identity"):
        encode_export_buffer_result(_export_result(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_worker_id", -1),
        ("buffer_id", 0),
        ("generation", 0),
    ],
)
def test_export_buffer_result_decode_rejects_invalid_live_identity(field, value):
    encoded = encode_export_buffer_result(_export_result())
    values = list(struct.unpack_from("<iQQ", encoded, 0))
    values[["owner_worker_id", "buffer_id", "generation"].index(field)] = value
    corrupted = struct.pack("<iQQ", *values) + encoded[20:]

    with pytest.raises(ValueError, match="live owner buffer identity"):
        decode_export_buffer_result(corrupted)


@pytest.mark.parametrize(
    "field,value",
    [
        ("importer_worker_id", -1),
        ("owner_worker_id", -1),
        ("buffer_id", 0),
        ("generation", 0),
    ],
)
def test_import_buffer_result_rejects_invalid_live_identity(field, value):
    with pytest.raises(ValueError, match="live imported buffer identity"):
        encode_import_buffer_result(_import_result(**{field: value}))


def test_served_frame_span_names_the_frame_the_caller_dispatched(monkeypatch):
    """The peer's span carries the header fields the caller's dispatch span does.

    Neither host can read the other's clock, so the two logs join on the frame
    header and on nothing else. The span brackets what this process did for
    that frame, which is the duration the caller's window has to hold.
    """
    emitted = []
    monkeypatch.setattr(remote_l3_session, "_host_spans_active", lambda: True)
    monkeypatch.setattr(remote_l3_session, "_monotonic_now_ns", lambda: 1_000)
    monkeypatch.setattr(remote_l3_session, "_emit_host_span", lambda *args: emitted.append(args))

    with remote_l3_session._served_frame_span(3, session_id=41, worker_id=2, sequence=17):
        pass

    assert len(emitted) == 1
    name, _inv, _hash, _depth, start_ns, _dur, attrs = emitted[0]
    assert name == "node.remote_task"
    assert start_ns == 1_000
    # Read back through the parser rather than compared to a literal: what the
    # peer writes has to be what `containment` reads, and a test spelling the
    # grammar a third time would let the writer and the reader drift apart
    # while both still passed their own assertions.
    assert attrs == containment.format_frame_key((41, 2, 17))


def test_served_frame_span_emits_nothing_while_host_spans_are_off(monkeypatch):
    """A gated-off run pays no formatting for a record it cannot write."""
    emitted = []
    monkeypatch.setattr(remote_l3_session, "_host_spans_active", lambda: False)
    monkeypatch.setattr(remote_l3_session, "_emit_host_span", lambda *args: emitted.append(args))

    with remote_l3_session._served_frame_span(3, session_id=41, worker_id=2, sequence=17):
        pass

    assert emitted == []
