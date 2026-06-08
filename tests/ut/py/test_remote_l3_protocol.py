# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import struct

from simpler.remote_l3_protocol import (
    CallableKind,
    RemoteRegistryTarget,
    decode_register_callable_command,
    decode_task_payload,
    encode_register_callable_command,
)


def test_task_payload_decode_preserves_scope_stats_config():
    prefix = b"/tmp/remote-scope"
    config = struct.pack("<iiiiiii", 7, 5, 0, 0, 0, 0, 1) + struct.pack("<I", len(prefix)) + prefix
    args = struct.pack("<III", 0, 0, 0)
    wire = (b"\xab" * 32) + config + args

    payload = decode_task_payload(wire)

    assert payload.config.block_dim == 7
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
