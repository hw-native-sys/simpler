# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L4 Host transfers over regions retained by their allocation owners."""

import asyncio
from dataclasses import dataclass


class TransferDomainError(RuntimeError):
    """Transfer failure with an explicit statement about possible outstanding access."""

    def __init__(self, message, *, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


@dataclass(frozen=True)
class TransferMember:
    role: str
    node_id: str
    boot: str
    registry: str

    def __post_init__(self):
        if self.role not in {"prefill", "decode"} or not all(
            isinstance(value, str) and value for value in self.identity
        ):
            raise ValueError("invalid transfer member")

    @property
    def identity(self):
        return self.node_id, self.boot, self.registry


@dataclass(frozen=True)
class TransferRegion:
    boot: str
    rank: int
    name: str
    extent: int
    handle: dict

    def __post_init__(self):
        if (
            not self.boot
            or type(self.rank) is not int
            or self.rank < 0
            or not self.name
            or type(self.extent) is not int
            or self.extent <= 0
        ):
            raise ValueError("invalid transfer region")


@dataclass(frozen=True)
class TransferCopy:
    source_rank: int
    source_region: str
    source_offset: int
    destination_rank: int
    destination_region: str
    destination_offset: int
    length: int

    def __post_init__(self):
        offsets = (self.source_rank, self.destination_rank, self.source_offset, self.destination_offset)
        if (
            any(type(value) is not int or value < 0 for value in offsets)
            or (type(self.length) is not int or self.length <= 0)
            or not all(isinstance(name, str) and name for name in (self.source_region, self.destination_region))
        ):
            raise ValueError("invalid transfer copy")


def _region_table(member, regions):
    table = {}
    for region in regions:
        key = region.rank, region.name
        if region.boot != member.boot or key in table:
            raise ValueError("transfer region identity mismatch or duplicate")
        table[key] = region
    if not table:
        raise ValueError("transfer member has no retained regions")
    return table


class TransferDomain:
    """A bound, directed Host transfer domain using an application control channel.

    Region descriptors refer to existing allocations. Callbacks run device
    operations in their owners' ACL contexts and leave the original buffers
    retained until a successful release. The submit callback must report the
    transfer ID only after every participating owner completes its native write.
    """

    def __init__(self, worker, source, destination, source_regions, destination_regions, *, connect, release):
        if source.role != "prefill" or destination.role != "decode" or source.identity == destination.identity:
            raise ValueError("transfer domain requires distinct Prefill and Decode members")
        self.source, self.destination = source, destination
        self._sources = _region_table(source, source_regions)
        self._destinations = _region_table(destination, destination_regions)
        self._source_ranks = {rank for rank, _ in self._sources}
        self._destination_ranks = {rank for rank, _ in self._destinations}
        self._worker, self._connect, self._release = worker, connect, release
        self._connected, self._transfers = set(), set()
        self._connecting = asyncio.Lock()
        self._closed = self._closing = self._retiring = self._uncertain = False

    def _check_active(self):
        if self._closed or self._retiring or self._uncertain:
            raise TransferDomainError("transfer domain is closed, retiring, or gated", uncertain=self._uncertain)

    async def ensure_connected(self, rank_pairs, *, prepare_destination):
        pairs = tuple(sorted(set(rank_pairs)))
        if not pairs or any(
            source not in self._source_ranks or destination not in self._destination_ranks
            for source, destination in pairs
        ):
            raise ValueError("transfer ranks are outside the bound domain")
        with self._worker._operation_lease("ensure_transfer_connected", requires_execution_context=False):
            async with self._connecting:
                self._check_active()
                try:
                    await prepare_destination(pairs)
                    self._check_active()
                    pending = tuple(pair for pair in pairs if pair not in self._connected)
                    if pending:
                        await self._connect(self.destination.identity, pending)
                        self._connected.update(pending)
                except BaseException as exc:
                    self._uncertain |= getattr(exc, "uncertain", not isinstance(exc, ValueError))
                    raise

    def validate_copies(self, copies):
        """Validate the entire batch against retained region identities and extents."""
        occupied = {}
        for copy in copies:
            source = self._sources.get((copy.source_rank, copy.source_region))
            destination = self._destinations.get((copy.destination_rank, copy.destination_region))
            if (
                source is None
                or destination is None
                or copy.source_offset + copy.length > source.extent
                or copy.destination_offset + copy.length > destination.extent
            ):
                raise ValueError("copy outside a bound transfer region")
            key = copy.destination_rank, copy.destination_region
            occupied.setdefault(key, []).append((copy.destination_offset, copy.destination_offset + copy.length))
        for intervals in occupied.values():
            intervals.sort()
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError("overlapping transfer destinations")

    async def put_sync_host(self, copies, *, transfer_id, submit):
        """Wait for the whole native batch before returning its transfer ID."""
        copies = tuple(copies)
        self.validate_copies(copies)
        if not copies or not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer requires a batch and transfer ID")
        with self._worker._operation_lease("put_transfer_sync_host", requires_execution_context=False):
            self._check_active()
            if transfer_id in self._transfers:
                raise ValueError("transfer ID is already active")
            if any((copy.source_rank, copy.destination_rank) not in self._connected for copy in copies):
                raise ValueError("transfer batch uses an unconnected rank pair")
            self._transfers.add(transfer_id)
            try:
                completed = await submit(copies, transfer_id)
                if completed != transfer_id:
                    raise TransferDomainError("transfer completion identity mismatch", uncertain=True)
                return completed
            except BaseException as exc:
                self._uncertain |= getattr(exc, "uncertain", not isinstance(exc, ValueError))
                raise
            finally:
                self._transfers.remove(transfer_id)

    async def close(self):
        """Stop admitting writes and retire the owners. Known release failures may be retried."""
        if self._closed:
            return
        with self._worker._operation_lease("close_transfer_domain", requires_execution_context=False):
            if self._uncertain or self._closing:
                raise TransferDomainError("transfer domain retirement is active or gated", uncertain=self._uncertain)
            if self._transfers or self._connecting.locked():
                raise TransferDomainError("transfer domain has active operations", uncertain=True)
            self._retiring = self._closing = True
            try:
                await self._release()
                self._closed = True
            except BaseException as exc:
                self._uncertain |= getattr(exc, "uncertain", not isinstance(exc, ValueError))
                raise
            finally:
                self._closing = False


def transfer_backend(transport):
    """Select the allocation-owner implementation for a Host transport."""
    if transport != "rdma":
        raise ValueError(f"unsupported Host transport: {transport}")
    from simpler.l4_pd_rdma import HixlTransferOwner  # noqa: PLC0415

    return HixlTransferOwner
