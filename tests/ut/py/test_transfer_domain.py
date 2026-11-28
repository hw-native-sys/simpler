# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host transfer identity, synchronization and lifetimes."""

import asyncio
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler import l4_pd_rdma
from simpler.transfer_domain import TransferCopy, TransferDomain, TransferDomainError, TransferMember, TransferRegion

COPY = TransferCopy(0, "keys", 8, 1, "cache", 16, 32)


def domain(connect, release):
    worker = SimpleNamespace(_operation_lease=lambda *args, **kwargs: nullcontext())
    return TransferDomain(
        worker,
        TransferMember("prefill", "p", "p-boot", "p-registry"),
        TransferMember("decode", "d", "d-boot", "d-registry"),
        (TransferRegion("p-boot", 0, "keys", 128, {}),),
        tuple(TransferRegion("d-boot", rank, "cache", 128, {}) for rank in (1, 2)),
        connect=connect,
        release=release,
    )


async def no_operation(*_args):
    pass


def test_write_waits_for_owner_and_reuses_connected_rank_pair():
    async def run():
        events = []
        started, finished = asyncio.Event(), asyncio.Event()

        async def prepare(pairs):
            events.append(("prepare", pairs))

        async def connect(peer, pairs):
            events.append(("connect", peer, pairs))

        async def submit(copies, transfer_id):
            assert copies == (COPY,)
            started.set()
            await finished.wait()
            return transfer_id

        async def release():
            events.append(("release",))

        transfer = domain(connect, release)
        with pytest.raises(ValueError, match="unconnected"):
            await transfer.put_sync_host((COPY,), transfer_id="request", submit=submit)
        for _ in range(2):
            await transfer.ensure_connected(((0, 1),), prepare_destination=prepare)
        assert events == [
            ("prepare", ((0, 1),)),
            ("connect", ("d", "d-boot", "d-registry"), ((0, 1),)),
            ("prepare", ((0, 1),)),
        ]
        pending = asyncio.create_task(transfer.put_sync_host((COPY,), transfer_id="request", submit=submit))
        started_wait = asyncio.create_task(started.wait())
        try:
            done, _ = await asyncio.wait((pending, started_wait), return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                await pending
                pytest.fail("submission returned before the owner finished")
        finally:
            started_wait.cancel()
            await asyncio.gather(started_wait, return_exceptions=True)
        assert not pending.done()
        with pytest.raises(TransferDomainError, match="active"):
            await transfer.close()
        finished.set()
        assert await pending == "request"
        await transfer.close()
        await transfer.close()
        assert events.count(("release",)) == 1

    asyncio.run(run())


def test_invalid_batch_is_rejected_before_any_owner_write():
    async def run():
        transfer = domain(no_operation, no_operation)
        await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)

        async def never_submit(*_args):
            pytest.fail("invalid batch reached an allocation owner")

        for invalid in (replace(COPY, destination_offset=120), replace(COPY, source_region="missing")):
            with pytest.raises(ValueError, match="outside"):
                await transfer.put_sync_host((COPY, invalid), transfer_id="request", submit=never_submit)
        with pytest.raises(ValueError, match="overlapping"):
            await transfer.put_sync_host((COPY, COPY), transfer_id="request", submit=never_submit)

    asyncio.run(run())


def test_unknown_completion_fences_release():
    async def run():
        async def wrong_completion(*_args):
            return "another-request"

        async def never_release():
            pytest.fail("an uncertain transfer released its domain")

        transfer = domain(no_operation, never_release)
        await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)
        with pytest.raises(TransferDomainError, match="identity mismatch") as error:
            await transfer.put_sync_host((COPY,), transfer_id="request", submit=wrong_completion)
        assert error.value.uncertain
        with pytest.raises(TransferDomainError, match="gated"):
            await transfer.close()

    asyncio.run(run())


@pytest.mark.parametrize("uncertain", [False, True])
def test_decode_prepare_failure_preserves_admission_only_if_known(uncertain):
    async def run():
        async def prepare(_pairs):
            raise TransferDomainError("prepare failed", uncertain=uncertain)

        transfer = domain(no_operation, no_operation)
        with pytest.raises(TransferDomainError, match="prepare failed"):
            await transfer.ensure_connected(((0, 1),), prepare_destination=prepare)
        if uncertain:
            with pytest.raises(TransferDomainError, match="gated"):
                await transfer.close()
        else:
            await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)
            await transfer.close()

    asyncio.run(run())


def test_prepare_rechecks_admission_after_inflight_write_becomes_uncertain():
    async def run():
        connections = []

        async def connect(_peer, pairs):
            connections.extend(pairs)

        transfer = domain(connect, no_operation)
        await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)

        async def submit(_copies, _transfer_id):
            raise TransferDomainError("write result unknown", uncertain=True)

        async def prepare(_pairs):
            with pytest.raises(TransferDomainError, match="write result unknown"):
                await asyncio.create_task(transfer.put_sync_host((COPY,), transfer_id="inflight", submit=submit))

        with pytest.raises(TransferDomainError, match="gated"):
            await transfer.ensure_connected(((0, 2),), prepare_destination=prepare)
        assert connections == [(0, 1)]

    asyncio.run(run())


def test_release_failure_does_not_reopen_the_domain():
    async def run():
        attempts = []

        async def release():
            attempts.append(None)
            if len(attempts) == 1:
                raise ValueError("peer retirement not acknowledged")

        transfer = domain(no_operation, release)
        await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)
        with pytest.raises(ValueError, match="retirement not acknowledged"):
            await transfer.close()
        with pytest.raises(TransferDomainError, match="retiring"):
            await transfer.put_sync_host((COPY,), transfer_id="late", submit=no_operation)
        with pytest.raises(TransferDomainError, match="retiring"):
            await transfer.ensure_connected(((0, 1),), prepare_destination=no_operation)
        await transfer.close()
        assert len(attempts) == 2

    asyncio.run(run())


def test_hixl_keeps_the_checked_ranges_from_a_one_pass_registry(monkeypatch):
    monkeypatch.setenv("PYPTO_HIXL_DEVICE_ENDPOINTS", '[["10.0.0.1", 0]]')
    monkeypatch.setenv("PYPTO_HIXL_PORT", "30000")
    client = Mock()
    client.remote_region.return_value = (0x2000, 128)
    monkeypatch.setattr(l4_pd_rdma, "HixlDeviceLink", Mock(return_value=client))
    source = l4_pd_rdma.HixlTransferOwner("192.0.2.1", "prefill")
    destination = l4_pd_rdma.HixlTransferOwner("192.0.2.2", "decode")
    local = source.register("p-boot", 0, "cache", 0x1000, 128)
    remote = destination.register("d-boot", 0, "cache", 0x2000, 128)
    peer = ("d", "d-boot", "d-registry")
    source.install(iter((remote,)), peer=peer)
    local.handle["address"] = 0x3000
    source.connect(peer, (0,))
    remote.handle["address"] = 0x4000
    source.put_sync_host(peer, (TransferCopy(0, "cache", 8, 0, "cache", 16, 32),))
    client.register.assert_called_once_with("cache", 0x1000, 128)
    client.remote_region.assert_called_once_with("cache")
    client.batch_put.assert_called_once_with([(0x1008, 0x2010, 32)])


def test_hixl_rejects_range_whose_end_wraps_uint64(monkeypatch):
    monkeypatch.setenv("PYPTO_HIXL_DEVICE_ENDPOINTS", '[["10.0.0.1", 0]]')
    monkeypatch.setenv("PYPTO_HIXL_PORT", "30000")
    source = l4_pd_rdma.HixlTransferOwner("192.0.2.1", "prefill")
    destination = l4_pd_rdma.HixlTransferOwner("192.0.2.2", "decode")
    with pytest.raises(ValueError, match="registration"):
        destination.register("d-boot", 0, "cache", (1 << 64) - 128, 128)
    source.register("p-boot", 0, "cache", 0x1000, 128)
    remote = destination.register("d-boot", 0, "cache", (1 << 64) - 129, 128)
    remote.handle["address"] += 1
    peer = ("d", "d-boot", "d-registry")
    with pytest.raises(ValueError, match="destination region"):
        source.install((remote,), peer=peer)
    with pytest.raises(ValueError, match="Decode regions"):
        source.install(iter(()), peer=peer)
