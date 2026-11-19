# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native import rollback and context restoration with a CPU-only ACL double."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires ELF interposition")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_import_cleanup_retains_its_owner_and_restores_context(tmp_path, stage):
    source = tmp_path / "acl.c"
    source.write_text(
        """
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
static void *context = (void *)1;
static int failure, counts[4];
int aclInit(const char *config) { return 0; }
int aclrtGetDevice(int *id) { *id = 4; return 0; }
int aclrtGetCurrentContext(void **ctx) { *ctx = context; return 0; }
int aclrtSetCurrentContext(void *ctx) { context = ctx; return 0; }
int aclrtSetDevice(int device) { abort(); }
int aclrtMemImportFromShareableHandle(uint64_t key, int device, void **out) { *out = (void *)0x55; return 0; }
int aclrtReserveMemAddress(void **out, size_t size, size_t align, void *hint, uint64_t flags) {
    *out = (void *)0x100000; return 0;
}
int aclrtMapMem(void *base, size_t size, size_t offset, void *physical, uint64_t flags) {
    return failure >= 2 ? 17 : 0;
}
int aclrtMemSetAccess(void *base, size_t size, const void *access, size_t count) { abort(); }
int close_stage(int stage) { ++counts[stage]; return failure == stage ? 7000 + stage : 0; }
int aclrtUnmapMem(void *base) { return close_stage(1); }
int aclrtReleaseMemAddress(void *base) { return close_stage(2); }
int aclrtFreePhysical(void *physical) { return close_stage(3); }
void fail_stage(int stage) { failure = stage; }
int close_calls(int stage) { return counts[stage]; }
"""
        + "\n".join(
            f"int {name}(void) {{ abort(); }}"
            for name in (
                "aclrtMemcpy",
                "aclrtMemGetAllocationGranularity",
                "aclrtMallocPhysical",
                "aclrtMemExportToShareableHandle",
            )
        )
    )
    library = tmp_path / "libascendcl.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    program = """
import ctypes, os, sys
import _task_interface as native
from simpler.buffer import wrap_device_malloc
from simpler.buffer_export import BufferExport, BufferMapping
stage = int(sys.argv[1])
driver = ctypes.CDLL(os.environ["LD_PRELOAD"])
root = wrap_device_malloc(0x200000, 4096, b"owner123", 1, "owner")
exported = BufferExport(native._encode_buffer_descriptor_wire(root.to_descriptor()), 1 << 63, 4, 23, 4096)
if stage:
    from simpler.buffer_export import BufferImportError
    driver.fail_stage(stage)
    try:
        if stage == 1:
            mapping = BufferMapping(exported)
            mapping.close()
        else:
            BufferMapping(exported)
    except BufferImportError as error:
        mapping = error.mapping
    except RuntimeError:
        assert stage == 1
    else:
        raise AssertionError("cleanup failure was not reported")
    driver.fail_stage(0)
    mapping.close()
    assert [driver.close_calls(i) for i in (1, 2, 3)] == [2 if stage == 1 else 0, 1 + (stage == 2), 1 + (stage == 3)]
else:
    mapping = BufferMapping(exported)
    driver.aclrtSetCurrentContext(ctypes.c_void_p(2))
    del mapping
    context = ctypes.c_void_p()
    driver.aclrtGetCurrentContext(ctypes.byref(context))
    assert context.value == 2, "implicit close changed caller context"
    assert [driver.close_calls(i) for i in (1, 2, 3)] == [1, 1, 1]
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(stage)],
        check=False,
        env=dict(os.environ, LD_PRELOAD=str(library)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
