# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private prepacked local NEXT_LEVEL submission for fixed rank bindings."""

from _task_interface import _snapshot_local_task_args

from .buffer import BackendKind
from .orchestrator import _admit_task_submission, _require_handle, _split_next_level_args


class PreparedLocalChipTask:
    """Own a validated snapshot without writable access to its TaskArgs."""

    def __init__(self, orch, handle, args, rank):
        worker = self._worker = orch._worker
        if worker.level != 3 or orch._expected_next_level_namespace() != "LOCAL_CHIP":
            raise ValueError("Prepared chip tasks require a local L3 Worker")
        self._handle = handle
        self._rank = rank
        c_args, sidecar = _split_next_level_args(args)
        self._args = _snapshot_local_task_args(c_args)
        if sidecar is not None:
            raise ValueError("Prepared chip tasks cannot carry sidecars")
        if any(
            self._args.tensor(i).buffer.backend_kind == BackendKind.REMOTE_SIDECAR
            for i in range(self._args.tensor_count())
        ):
            raise ValueError("Prepared chip tasks cannot carry remote allocation descriptors")
        self._identities = frozenset(self._args.identities())

    def submit(self, orch, cfg):
        worker = self._worker
        if orch._worker is not worker or worker._building_run_resources is None:
            raise RuntimeError("Prepared chip task requires its owning Worker callback")
        rank = self._rank
        digest, kind, namespace, eligible = _require_handle(
            self._handle, kind="orch.submit_next_level", worker=worker, expected_namespace="LOCAL_CHIP"
        )
        worker._require_local_next_level_target(rank, api="submit_next_level")
        worker._acquire_child_host_args(self._args, rank, device_target=True)
        with worker._child_prov_lock:
            worker._child_prov_check_dispatch_locked(self._args, rank, api="submit_next_level")
            worker._building_run_resources.touched_identities.update(self._identities)
            _admit_task_submission(worker)
            return orch._o.submit_next_level(digest, kind, namespace, self._args, cfg, rank, eligible, None)
