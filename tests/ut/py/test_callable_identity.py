# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import cast

import pytest
from simpler import callable_identity
from simpler.callable_identity import (
    CallableHandle,
    CallableKindName,
    TargetNamespaceName,
    build_chip_callable_descriptor,
    build_python_import_descriptor,
    build_python_serialized_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
    parse_python_import_target,
    validate_hashid,
)
from simpler.task_interface import ChipCallable
from simpler.worker import Worker, _pack_py_callable_payload


def _py_target(args):
    return args


def test_python_descriptor_hash_is_stable_for_same_serialized_payload():
    payload = _pack_py_callable_payload(_py_target)
    descriptor = build_python_serialized_descriptor(payload)

    hashid = compute_callable_hashid(descriptor)

    assert hashid == compute_callable_hashid(build_python_serialized_descriptor(payload))
    assert len(hashid_to_digest(hashid)) == 32


def test_chip_descriptor_changes_when_callable_blob_changes():
    first = ChipCallable.build(signature=[], func_name="x", binary=b"\x01", children=[])
    second = ChipCallable.build(signature=[], func_name="y", binary=b"\x02", children=[])

    assert compute_callable_hashid(build_chip_callable_descriptor(target=first)) != compute_callable_hashid(
        build_chip_callable_descriptor(target=second)
    )


def test_python_import_descriptor_hash_is_stable():
    module, qualname = parse_python_import_target("pkg.mod:Class.method")
    descriptor = build_python_import_descriptor(module, qualname)

    assert (module, qualname) == ("pkg.mod", "Class.method")
    assert compute_callable_hashid(descriptor) == compute_callable_hashid(
        build_python_import_descriptor("pkg.mod", "Class.method")
    )
    assert len(hashid_to_digest(compute_callable_hashid(descriptor))) == 32


@pytest.mark.parametrize(
    "target",
    [
        "pkg.mod.fn",
        ":fn",
        "pkg:",
        ".pkg:fn",
        "pkg..mod:fn",
        "pkg.mod:Class..method",
        "pkg.mod:<locals>.fn",
        "pkg.mod:bad-name",
        object(),
    ],
)
def test_python_import_target_validation_rejects_invalid_targets(target):
    with pytest.raises((TypeError, ValueError)):
        parse_python_import_target(target)


@pytest.mark.parametrize("hashid", ["", "sha256:ABC", "md5:" + "0" * 64, "sha256:" + "0" * 63])
def test_hashid_validation_rejects_noncanonical_values(hashid):
    with pytest.raises(ValueError, match="HASHID_FORMAT_INVALID"):
        validate_hashid(hashid)


def test_callable_identity_public_exports_do_not_include_worker_state():
    assert "CallableHandle" in callable_identity.__all__
    assert "_CallableIdentityState" not in callable_identity.__all__


def test_worker_register_returns_opaque_handle_and_deduplicates_same_identity():
    worker = Worker(level=3, num_sub_workers=0)
    try:
        first = worker.register(_py_target)
        second = worker.register(_py_target)

        assert isinstance(first, CallableHandle)
        assert not isinstance(first, int)
        assert first.hashid == second.hashid
        assert first.digest == second.digest
        assert first._handle_id != second._handle_id
        assert worker._identity_registry[first.digest].slot_id >= 0
        assert len(worker._callable_registry) == 1
        assert worker._identity_registry[first.digest].ref_count == 2
    finally:
        worker.close()


def test_callable_handle_public_constructor_returns_unbound_handle():
    handle = CallableHandle(
        "sha256:" + "0" * 64,
        cast(CallableKindName, "PYTHON_SERIALIZED"),
        cast(TargetNamespaceName, "LOCAL_PYTHON"),
    )

    assert handle.digest == bytes(32)
    assert handle._handle_id == -1
    assert handle._owner_id is None
    assert not hasattr(handle, "slot_id")
    assert not hasattr(handle, "cid")


def test_callable_handle_accepts_remote_dispatcher_import_namespace():
    descriptor = build_python_import_descriptor("pkg.mod", "remote_entry")
    handle = CallableHandle(
        compute_callable_hashid(descriptor),
        cast(CallableKindName, "PYTHON_IMPORT"),
        cast(TargetNamespaceName, "REMOTE_TASK_DISPATCHER"),
    )

    assert handle.kind == "PYTHON_IMPORT"
    assert handle.target_namespace == "REMOTE_TASK_DISPATCHER"
    assert handle.digest == hashid_to_digest(handle.hashid)


def test_forged_public_handle_is_rejected_by_worker_apis():
    worker = Worker(level=3, num_sub_workers=0)
    real = worker.register(_py_target)
    forged = CallableHandle(
        real.hashid,
        cast(CallableKindName, real.kind),
        cast(TargetNamespaceName, real.target_namespace),
    )
    try:
        with pytest.raises(KeyError, match="does not belong|not live"):
            worker.unregister(forged)
        with pytest.raises(KeyError, match="does not belong|not live"):
            worker._resolve_handle(forged)
    finally:
        worker.close()


def test_mutated_handle_fields_are_rejected():
    worker = Worker(level=3, num_sub_workers=0)
    handle = worker.register(_py_target)
    try:
        handle.kind = "CHIP_CALLABLE"
        with pytest.raises(RuntimeError, match="CALLABLE_HANDLE_MUTATED"):
            worker._resolve_handle(handle)
    finally:
        worker.close()


def test_uncertain_cleanup_hashid_blocks_live_handle_resolution():
    worker = Worker(level=3, num_sub_workers=0)
    handle = worker.register(_py_target)
    try:
        worker._uncertain_hashids.add(handle.digest)
        with pytest.raises(RuntimeError, match="REGISTER_CLEANUP_UNCERTAIN"):
            worker._resolve_handle(handle)
    finally:
        worker.close()
