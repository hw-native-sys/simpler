# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep kernel submission within the v9 capture-safe operation vocabulary."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    "src/common/platform/onboard/host/kernel_launch_binder.cpp",
    "src/common/platform/onboard/host/kernel_launch_native.cpp",
    "src/common/platform/onboard/host/kernel_launch_sequence.h",
)
ALLOWED_ACL = {
    "aclrtQueryEventStatus",
    "aclrtStreamWaitEvent",
    "aclrtRecordEvent",
    "aclrtMemsetAsync",
    "aclrtLaunchKernel",
    "aclrtLaunchKernelWithHostArgs",
}


def violations(source):
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    calls = set(re.findall(r"\b((?:aclrt|acl|rt)[A-Z]\w*)\s*\(", code))
    banned = re.findall(
        r"\b(?:malloc|calloc|realloc|free|make_graph_host_args|submit_graph_template)\s*\("
        r"|\b(?:new|delete)\b|std::(?:vector|make_unique|make_shared)\b",
        code,
    )
    return sorted(calls - ALLOWED_ACL) + banned


@pytest.mark.parametrize("path", SOURCES)
def test_kernel_submission_has_no_allocation_sync_or_capture_query(path):
    assert not violations((ROOT / path).read_text())


def test_binder_has_one_enqueue_composition():
    source = (ROOT / SOURCES[0]).read_text()
    assert len(re.findall(r"\benqueue_kernel_launch_sequence\s*\(", source)) == 1


@pytest.mark.parametrize(
    "operation",
    [
        "aclrtSynchronizeStream(stream);",
        "rtStreamAddToModel(stream, model);",
        "rtStreamSynchronize(stream);",
        "rtMalloc(&ptr, bytes, policy);",
        "aclInit(nullptr);",
        "aclFinalize();",
        "aclrtStreamGetCaptureInfo(stream);",
        "aclrtMalloc(&ptr, bytes, policy);",
        "aclrtCreateEvent(&event);",
        "aclrtDestroyStream(stream);",
        "aclrtModelExecuteAsync(model, stream);",
        "malloc(bytes);",
        "new int(1);",
        "std::vector<int> data;",
        "make_graph_host_args(source, args);",
    ],
)
def test_guard_rejects_forbidden_submission_operations(operation):
    assert violations(operation)


def test_submission_module_does_not_import_unmerged_context_or_hbg_interfaces():
    paths = (
        *SOURCES,
        "src/common/platform/include/host/kernel_launch_binder.h",
        "src/common/platform/onboard/host/kernel_launch_native.h",
    )
    forbidden = (
        "kernel_execution_state.h",
        "kernel_device_resources.h",
        "kernel_invocation_validation.h",
        "host_build_graph/",
    )
    for path in paths:
        includes = re.findall(r'^#include\s+[<"]([^>"]+)', (ROOT / path).read_text(), re.M)
        assert not [include for include in includes if any(item in include for item in forbidden)], path
