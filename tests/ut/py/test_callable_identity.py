# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from simpler.callable_identity import (
    CallableHandle,
    build_chip_callable_descriptor,
    build_python_serialized_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
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


@pytest.mark.parametrize("hashid", ["", "sha256:ABC", "md5:" + "0" * 64, "sha256:" + "0" * 63])
def test_hashid_validation_rejects_noncanonical_values(hashid):
    with pytest.raises(ValueError, match="HASHID_FORMAT_INVALID"):
        validate_hashid(hashid)


def test_worker_register_returns_opaque_handle_and_deduplicates_same_identity():
    worker = Worker(level=3, num_sub_workers=0)
    try:
        first = worker.register(_py_target)
        second = worker.register(_py_target)

        assert isinstance(first, CallableHandle)
        assert not isinstance(first, int)
        assert first.hashid == second.hashid
        assert first.digest == second.digest
        assert first != second
        assert first != worker._identity_registry[first.digest].slot_id
        assert len(worker._callable_registry) == 1
        assert worker._identity_registry[first.digest].ref_count == 2
    finally:
        worker.close()


def test_callable_handle_public_constructor_does_not_expose_slot_identity():
    handle = CallableHandle("sha256:" + "0" * 64, "PYTHON_SERIALIZED", "LOCAL_PYTHON")

    assert handle.digest == bytes(32)
    assert not hasattr(handle, "_slot_id")


def test_forged_public_handle_is_rejected_by_worker_apis():
    worker = Worker(level=3, num_sub_workers=0)
    real = worker.register(_py_target)
    forged = CallableHandle(real.hashid, real.kind, real.target_namespace)
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
