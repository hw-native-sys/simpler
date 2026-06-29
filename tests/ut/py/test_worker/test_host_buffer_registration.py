# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side reachability classifier for post-fork host buffers.

Error path C: a host tensor created *after* the chip children fork is invisible
to them unless ``register_host_buffer`` maps it in. ``submit_next_level`` calls
``_stage_host_buffers_for_chip_submit`` *before any dispatch*; for an
unregistered post-fork tensor it must raise an actionable error rather than
silently submit a pointer the child cannot read.

These are pure host-side unit tests: the classifier depends only on the fork
snapshot (``_fork_maps``) and ``/proc/self/maps``, so we inject that state
directly and never fork, compile a kernel, or touch a device (mirrors the
white-box style of ``test_host_worker.py``). The end-to-end registered
round-trip (mechanism B) lives in the a2a3sim scene test.
"""

from __future__ import annotations

import ctypes
import warnings

import pytest
import torch
from simpler.task_interface import TaskArgs, TensorArgType
from simpler.worker import Worker, _HostBufEntry, _read_self_maps, _va_lookup_inode

from simpler_setup import make_tensor_arg

_SIZE = 128 * 128
# Element count whose float32 allocation (64 MiB) is above glibc's max dynamic
# mmap threshold (32 MiB on 64-bit), so it is always its own anonymous mmap (a
# fresh post-fork VA), never served from a fork-time heap arena.
_BIG_ELEMS = 16 * 1024 * 1024

# Tests that inject a *real* fork snapshot (or read a tensor's real inode) only
# make sense where /proc/self/maps exists. On a non-procfs platform (macOS CI)
# _read_self_maps() is [], so skip them; the no-procfs behaviour itself is
# covered by test_passes_through_without_procfs.
_requires_procfs = pytest.mark.skipif(not _read_self_maps(), reason="/proc/self/maps unavailable")


def _classifier_worker(*, fork_maps):
    """A ``Worker`` with only the classifier state populated.

    ``_stage_host_buffers_for_chip_submit`` reads nothing else, so we bypass the
    real init/fork and inject the fork snapshot (sorted ``(lo, hi, inode)``
    ranges, or ``None`` for "no procfs") by hand.
    """
    w = Worker.__new__(Worker)
    w._host_buf_registry = {}
    w._host_buf_sorted_ptrs = []
    w._fork_maps = fork_maps
    w._submit_maps = None
    w._pending_host_copyback = []
    w._warned_no_procfs_passthrough = False
    return w


def _one_input_arg(tensor):
    ta = TaskArgs()
    ta.add_tensor(make_tensor_arg(tensor), TensorArgType.INPUT)
    return ta


class TestErrorPathCRejection:
    """Unregistered post-fork tensors are refused at submit, before dispatch."""

    def test_post_fork_anonymous_tensor_rejected(self):
        # inode 0 (anonymous mmap); VA not in the (empty) fork snapshot.
        w = _classifier_worker(fork_maps=[])
        anon = torch.zeros(_BIG_ELEMS, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="register_host_buffer"):
            w._stage_host_buffers_for_chip_submit(_one_input_arg(anon))

    def test_post_fork_shared_tensor_rejected(self):
        # File-backed (share_memory_) tensor whose inode was not inherited at fork.
        w = _classifier_worker(fork_maps=[])
        shared = torch.zeros(_SIZE, dtype=torch.float32).share_memory_()
        with pytest.raises(RuntimeError, match="register_host_buffer"):
            w._stage_host_buffers_for_chip_submit(_one_input_arg(shared))

    @_requires_procfs
    def test_fork_inherited_tensor_passes(self):
        # The legitimate common case: a tensor whose real mapping is in the fork
        # snapshot (so the child inherited the exact VA + inode) is accepted
        # without raising. Guards against the coverage check over-rejecting.
        t = torch.zeros(_SIZE, dtype=torch.float32)
        w = _classifier_worker(fork_maps=_read_self_maps())
        w._stage_host_buffers_for_chip_submit(_one_input_arg(t))  # no raise

    @_requires_procfs
    def test_same_inode_other_va_rejected(self):
        # Regression for the inode-only gap: a file-backed tensor whose inode WAS
        # present at fork but at a *different* VA (a post-fork re-mmap of the same
        # file) must still be rejected — the child never inherited this VA, so
        # matching the inode alone is not enough. The old inode-membership check
        # accepted it; the range-coverage check rejects it.
        shared = torch.zeros(_SIZE, dtype=torch.float32).share_memory_()
        addr = shared.data_ptr()
        inode = _va_lookup_inode(_read_self_maps(), addr)
        assert inode and inode != 0, "share_memory_ tensor must be file-backed"
        # Same inode, but a VA range entirely below the tensor — does not cover it.
        fork_maps = [(addr - 0x200000, addr - 0x100000, inode)]
        w = _classifier_worker(fork_maps=fork_maps)
        with pytest.raises(RuntimeError, match="register_host_buffer"):
            w._stage_host_buffers_for_chip_submit(_one_input_arg(shared))

    def test_passes_through_without_procfs(self):
        # _fork_maps is None => no fork snapshot (no procfs, e.g. macOS):
        # reachability cannot be classified, so an unregistered tensor is passed
        # through unvalidated rather than rejected — a fork-inherited tensor is the
        # legitimate common case and must keep working. Error path C is only
        # enforced where procfs exists. The pass-through emits a one-time warning so
        # the caller knows visibility went unverified. Regression guard for macOS.
        w = _classifier_worker(fork_maps=None)
        anon = torch.zeros(_BIG_ELEMS, dtype=torch.float32)
        with pytest.warns(UserWarning, match="visibility cannot be validated"):
            w._stage_host_buffers_for_chip_submit(_one_input_arg(anon))  # no raise
        # Latched: a second pass-through is silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            w._stage_host_buffers_for_chip_submit(_one_input_arg(anon))


def _registered_worker(entry):
    """A ``Worker`` with only the registered-buffer staging state populated.

    The registered path in ``_stage_host_buffers_for_chip_submit`` returns before
    the fork-snapshot check, so it reads only ``_host_buf_registry``,
    ``_host_buf_sorted_ptrs``, ``_staged_output_ranges`` and
    ``_pending_host_copyback``.
    """
    w = Worker.__new__(Worker)
    w._host_buf_registry = {entry.data_ptr: entry}
    w._host_buf_sorted_ptrs = [entry.data_ptr]
    w._staged_output_ranges = []
    w._pending_host_copyback = []
    return w


def _entry_for(parent, shm_buf):
    """A registered entry mapping ``parent``'s VA onto a caller-owned ``shm_buf``
    (a ctypes buffer standing in for the child-visible shm)."""
    return _HostBufEntry(
        token=1,
        data_ptr=parent.data_ptr(),
        nbytes=parent.numel() * parent.element_size(),
        shm=None,  # type: ignore[arg-type]  # staging reads only shm_base
        shm_name="",
        shm_base=ctypes.addressof(shm_buf),
        tensor=parent,
    )


def _arg(tensor, tag):
    ta = TaskArgs()
    ta.add_tensor(make_tensor_arg(tensor), tag)
    return ta


class TestInRunProducerConsumer:
    """Copy-in must not clobber a buffer an earlier task in the same run wrote.

    The registered shm is the live child↔child medium: producer task A writes its
    device output into the shm, the OverlapMap orders consumer task B's read after
    it, and B's host-side copy-in (stale parent → shm) is the only thing that can
    race that output. Submit order == dependency order, so by the time B is staged
    A's range is already recorded.
    """

    def test_consumer_input_does_not_clobber_producer_output(self):
        # Stale parent = 7.0; shm pre-filled with 3.0 stands in for A's device
        # output already mirrored into the shm.
        parent = torch.full((_SIZE,), 7.0, dtype=torch.float32)
        shm_buf = (ctypes.c_float * _SIZE)(*([3.0] * _SIZE))
        w = _registered_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.OUTPUT))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.INPUT))
        # copy-in skipped: the producer's output survives (3.0), not stale 7.0.
        assert list(shm_buf[:4]) == [3.0, 3.0, 3.0, 3.0]

    def test_plain_input_still_copies_in(self):
        # No earlier task wrote the range, so the parent snapshot must reach the shm.
        parent = torch.full((_SIZE,), 7.0, dtype=torch.float32)
        shm_buf = (ctypes.c_float * _SIZE)(*([3.0] * _SIZE))
        w = _registered_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.INPUT))
        assert list(shm_buf[:4]) == [7.0, 7.0, 7.0, 7.0]

    def test_no_dep_over_produced_range_raises(self):
        # NO_DEP skips the OverlapMap, so the read is unordered against the
        # in-run producer — neither copy-in nor skip is safe.
        parent = torch.full((_SIZE,), 7.0, dtype=torch.float32)
        shm_buf = (ctypes.c_float * _SIZE)()
        w = _registered_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.OUTPUT))
        with pytest.raises(RuntimeError, match="NO_DEP"):
            w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.NO_DEP))

    def test_partial_overlap_raises(self):
        # Producer writes [0, 64); consumer reads [32, 96) — straddles the
        # boundary, so neither a full copy-in nor a full skip is correct.
        parent = torch.full((256,), 7.0, dtype=torch.float32)
        shm_buf = (ctypes.c_float * 256)()
        w = _registered_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent[0:64], TensorArgType.OUTPUT))
        with pytest.raises(RuntimeError, match="partially overlaps"):
            w._stage_host_buffers_for_chip_submit(_arg(parent[32:96], TensorArgType.INPUT))


class TestReadSelfMaps:
    def test_returns_empty_when_procfs_absent(self, monkeypatch):
        # _read_self_maps must degrade to [] (not FileNotFoundError) where
        # /proc/self/maps does not exist — the bug that broke the macOS jobs.
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/self/maps":
                raise FileNotFoundError(2, "No such file or directory", "/proc/self/maps")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert _read_self_maps() == []
