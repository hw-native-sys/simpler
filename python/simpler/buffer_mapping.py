# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Package-internal Host ownership and explicit per-chip mapping implementation.

The supported user surface is Buffer/Tensor plus Worker methods;
this module's resource and grant-registration helpers are not public APIs.
"""

from __future__ import annotations

from .buffer import (
    AccessMode,
    Tensor,
)


class MappingResource:
    """Routing for issued references; the target ImportRegistry owns grants."""

    def __init__(self, buffer, workers, view_scope, view_control, *, identity=None):
        self.buffer = buffer
        self.identity = buffer.identity if buffer is not None else identity
        self._descriptor = buffer.to_descriptor() if buffer is not None else None
        self.workers = tuple(workers)
        self.closed = False
        self._view_scope = view_scope
        self._view_control = view_control
        self._next_reference = 1
        self._mapping_targets = {}
        self._revoked_refs = set()

    def _check_source(self, tensor):
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host mapping requires the live source backing")
        if tensor.buffer != self._descriptor:
            raise ValueError("Host Tensor differs from its source registration")

    def map_view(self, tensor: Tensor, worker: int, *, access=AccessMode.READWRITE) -> Tensor:
        with self._view_scope(self, worker, "map_view"):
            self._check_source(tensor)
            if tensor.mapping_ref:
                raise ValueError("map_view requires the native Tensor view")
            access = AccessMode(access)
            if worker not in self.workers:
                raise ValueError("Host mapping requires a local device endpoint")
            # An unacknowledged map has no caller-visible Tensor. Retry its
            # cleanup through this public operation before issuing another ref.
            for ref, route in tuple(self._mapping_targets.items()):
                if route == (worker, False):
                    self._view_control(self, worker, ref, 0, unmap=True)
                    self._mapping_targets.pop(ref)
            if any(self._mapping_targets.get(ref) == (worker, True) for ref in self._revoked_refs):
                raise RuntimeError("Host mapping cleanup is unconfirmed")
            if self._next_reference > 255:
                raise RuntimeError("Host mapping references exhausted; allocate a new backing")
            reference = self._next_reference
            self._next_reference += 1
            mapped = tensor._with_mapping_ref(reference)
            self._mapping_targets[reference] = (worker, False)
            self._revoked_refs.add(reference)
            # Publish no authority until the target confirms its restricted record.
            try:
                self._view_control(self, worker, reference, access, unmap=False)
                self._revoked_refs.remove(reference)
                self._mapping_targets[reference] = (worker, True)
                return mapped
            except BaseException:
                self._revoked_refs.add(reference)
                self._mapping_targets[reference] = (worker, False)
                raise

    def unmap_view(self, tensor: Tensor):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        route = self._mapping_targets.get(reference)
        if route is None:
            if reference in self._revoked_refs:
                return
            raise ValueError("Host Tensor has no issued mapping")
        worker, _acknowledged = route
        with self._view_scope(self, worker, "unmap_view"):
            self._check_source(tensor)
            self._revoked_refs.add(reference)
            self._view_control(self, worker, reference, 0, unmap=True)
            self._mapping_targets.pop(reference, None)

    def validate(self, tensor, worker, *, native=False):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        if worker not in self.workers or (
            not (native and reference == 0)
            and (reference in self._revoked_refs or self._mapping_targets.get(reference) != (worker, True))
        ):
            raise ValueError("Host Tensor requires a live explicit mapping for the target worker")
