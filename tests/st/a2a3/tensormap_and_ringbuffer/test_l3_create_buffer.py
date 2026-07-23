#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 create_buffer export handshake (P1-B 3c-2a).

``Worker.create_buffer`` allocates a born-shared BufferHandle and broadcasts its typed
BufferHandleDescriptor to every forked child via ``_CTRL_REGISTER_HANDLE``; each child decodes it
and registers ``canonical identity -> local base`` in its ImportRegistry. A create_buffer that
returns without raising means the chip child successfully decoded the 216-byte descriptor and mapped
the POSIX shm (a broadcast error would have propagated). This exercises the owner export + child
import handshake end-to-end; wiring the handle into task dispatch (BufferRef) is 3c-2b.

a2a3sim: create_buffer is pure host-side (POSIX shm + a control broadcast to the forked chip child),
no platform branching. The vector callable exists only to give the L3 worker a chip child to fork.
"""

from simpler.buffer_handle import AddressSpace, BackendKind, Visibility
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3CreateBufferHandshake(SceneTestCase):
    """create_buffer owner-export + child-import handshake on a single L3 worker."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
        ],
    }

    CASES = [
        {"name": "create_buffer_handshake", "platforms": ["a2a3sim"]},
    ]

    def test_run(self, st_worker):
        worker = st_worker

        # Each create_buffer exports a descriptor to the forked chip child; a child-side decode or
        # shm-map failure would surface here as a raised RuntimeError.
        h0 = worker.create_buffer(256)
        h1 = worker.create_buffer(512)

        # Owner-side handle contract.
        assert h0.backend_kind == BackendKind.POSIX_SHM
        assert h0.address_space == AddressSpace.HOST
        assert h0.visibility == Visibility.SHARED
        assert len(h0.identity.owner_instance_id) == 16
        assert h0.identity.owner_instance_id == h1.identity.owner_instance_id  # same incarnation
        assert h0.identity.buffer_id != h1.identity.buffer_id  # monotonic
        assert h0.identity.generation == 1
        assert h0.nbytes == 256 and h1.nbytes == 512

        # The backing is usable in place (born-shared shm).
        assert h0.shm is not None
        buf = h0.shm.buf
        assert buf is not None
        buf[:4] = b"\xde\xad\xbe\xef"
        assert bytes(buf[:4]) == b"\xde\xad\xbe\xef"

        # Worker close (framework-driven) releases both handles via _release_all_buffer_handles.
