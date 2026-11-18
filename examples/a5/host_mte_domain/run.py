#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous A5 peer Host reads with retained communication domains."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from simpler.task_interface import AddressSpace, CommBufferSpec, DataType, TaskArgs, TensorArgType

if __package__:
    from .device_gate import release_domain_during_reads
else:
    from device_gate import release_domain_during_reads

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[2] / "build/host_mte_domain"
RUNTIMES = ("tensormap_and_ringbuffer", "host_build_graph", "percounter")
COUNT = 16384
NBYTES = COUNT * 4
PREFIX_BYTES = 256
DOMAIN_MEMBERS = {"forward": (0, 1), "reverse": (1, 0), "device": (0, 1)}


def _input_key(name, worker_id, generation):
    return f"{name}_{worker_id}_{generation}"


def generate_inputs():
    return {
        _input_key(name, worker_id, generation): (
            np.arange(COUNT, dtype=np.float32)
            + np.float32(domain_id * 1000000 + worker_id * 20000 + generation * 100000)
        )
        for domain_id, name in enumerate(DOMAIN_MEMBERS)
        for worker_id in (0, 1)
        for generation in (1, 2)
    }


def _source_digest():
    digest = hashlib.sha256()
    sources = [
        Path(__file__).resolve(),
        ROOT / "device_gate.py",
        ROOT / "test_host_mte_domain.py",
        *sorted((ROOT / "kernels").rglob("*.cpp")),
    ]
    for source in sources:
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    digest.update((ROOT.parents[2] / "pto_isa.pin").read_bytes())
    return digest.hexdigest()


def prepare_inputs(runtime, output):
    """Save test data during the scene-test precompile pass."""
    target = Path(output) / runtime
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "inputs.npz", **generate_inputs())
    metadata = {
        "runtime": runtime,
        "machine": platform.machine(),
        "sources": _source_digest(),
        "inputs": hashlib.sha256((target / "inputs.npz").read_bytes()).hexdigest(),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PREPARED runtime={runtime} artifacts={target}", flush=True)


def load_inputs(runtime, output):
    target = Path(output) / runtime
    metadata = json.loads((target / "metadata.json").read_text())
    if (metadata["runtime"], metadata["machine"], metadata["sources"]) != (
        runtime,
        platform.machine(),
        _source_digest(),
    ):
        raise ValueError("prepared MTE inputs do not match this runtime, host or source; rerun scene_test_compile")
    if hashlib.sha256((target / "inputs.npz").read_bytes()).hexdigest() != metadata["inputs"]:
        raise ValueError("prepared input checksum mismatch")
    with np.load(target / "inputs.npz", allow_pickle=False) as archive:
        inputs = dict(archive)
    return inputs


def _specs():
    return [
        CommBufferSpec("reserved", "uint8", PREFIX_BYTES, PREFIX_BYTES),
        CommBufferSpec("payload", "float32", COUNT, NBYTES),
    ]


def _write_host(domain, name, generation, inputs):
    offset, nbytes = domain.buffer_range("payload")
    for rank in (0, 1):
        member = domain.member(rank)
        payload = inputs[_input_key(name, member.local_worker_id, generation)]
        with member.buffers["payload"].shm.buf[offset : offset + nbytes] as memory:
            memory[:] = payload.tobytes()


def run_lifecycle(worker, entry, runtime, inputs, config):  # noqa: PLR0915 -- one retained-domain lifecycle
    domains, outputs, initialized = {}, {}, []
    scratch = worker.create_buffer(NBYTES)
    try:
        for name in DOMAIN_MEMBERS:
            for worker_id in (0, 1):
                outputs[name, worker_id] = [
                    worker.alloc_child_tensor(worker_id, (COUNT,), DataType.FLOAT32) for _ in range(2)
                ]

        def allocate(orch, name):
            def initialize(domain):
                _write_host(domain, name, 1, inputs)
                initialized.append((name, domain.domain_id))

            domain = orch.get_or_allocate_domain(
                name=name,
                members=DOMAIN_MEMBERS[name],
                window_size=PREFIX_BYTES + NBYTES,
                buffers=_specs(),
                memory_source=AddressSpace.DEVICE if name == "device" else AddressSpace.HOST,
                workspace_engines=(),
                retain_after_run=True,
                initializer=initialize if name != "device" else None,
            )
            if name in domains:
                assert domain is domains[name]
            else:
                domains[name] = domain
                if name == "device":
                    for rank in (0, 1):
                        orch.copy_to_domain(domain, rank, inputs[_input_key(name, rank, 1)].tobytes(), buffer="payload")
            assert domain.buffer_range("payload") == (PREFIX_BYTES, NBYTES)
            return domain

        def submit(orch, _args, config, *, names, gates=()):
            for name in names:
                allocate(orch, name)
            for name in names:
                domain = domains[name]
                for rank in (0, 1):
                    member = domain.member(rank)
                    worker_id = member.local_worker_id
                    args = TaskArgs()
                    args.add_tensor(member.buffers["payload"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.INPUT)
                    for output in outputs[name, worker_id]:
                        args.add_tensor(output.tensor((COUNT,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                    args.add_scalar(domain.device_context(worker_id))
                    args.add_scalar(gates[worker_id].address if gates and name == "forward" else 0)
                    orch.submit_next_level(entry, args, config, worker=worker_id)

        def clear(names):
            assert scratch.shm is not None and scratch.shm.buf is not None
            with scratch.shm.buf[:] as memory:
                memory[:] = bytes(NBYTES)
            for name in names:
                for worker_id in (0, 1):
                    for output in outputs[name, worker_id]:
                        worker.copy_to(output, scratch)

        def verify(names, generation):
            for name in names:
                for worker_id in (0, 1):
                    peer = 1 - worker_id
                    expected = inputs[_input_key(name, peer, 1 if name == "device" else generation)]
                    for increment, output in enumerate(outputs[name, worker_id]):
                        worker.copy_from(scratch, output)
                        assert scratch.shm is not None and scratch.shm.buf is not None
                        with scratch.shm.buf[:] as memory:
                            actual = np.frombuffer(memory, dtype=np.float32).copy()
                        np.testing.assert_array_equal(actual, expected + increment)
            print(f"PASS runtime={runtime} generation={generation} domains={names} peer_and_consumer=True", flush=True)

        names = tuple(DOMAIN_MEMBERS)
        identities = None
        for generation in (1, 2):
            if generation == 2:
                for name in ("forward", "reverse"):
                    _write_host(domains[name], name, generation, inputs)
            clear(names)
            worker.run(lambda orch, args, config: submit(orch, args, config, names=names), config=config)
            current = tuple(
                (
                    domains[name].domain_id,
                    tuple(domains[name].local_context(rank).local_window_base for rank in (0, 1)),
                )
                for name in names
            )
            if identities is None:
                identities = current
            assert current == identities
            assert initialized == [(name, domains[name].domain_id) for name in ("forward", "reverse")]
            verify(names, generation)

        saved_mapping = domains["forward"].member(0).buffers["payload"].shm
        clear(names)
        release_domain_during_reads(
            worker,
            domains["forward"],
            lambda gates: worker.submit(
                lambda orch, args, config: submit(orch, args, config, names=names, gates=gates), config=config
            ),
        )
        assert domains["forward"].freed and saved_mapping.buf is None
        verify(names, 2)
        print(f"PASS runtime={runtime} release_while_peer_reads_gated=True", flush=True)

        names = ("reverse", "device")
        clear(names)
        worker.run(lambda orch, args, config: submit(orch, args, config, names=names), config=config)
        verify(names, 2)
        domains["reverse"].release()
        assert domains["reverse"].released and domains["reverse"].freed

        clear(("device",))
        worker.run(lambda orch, args, config: submit(orch, args, config, names=("device",)), config=config)
        verify(("device",), 2)
        domains["device"].release()
        assert domains["device"].freed
        for buffers in outputs.values():
            for output in buffers:
                worker.free(output)
        print(f"PASS runtime={runtime} retained_updates_reordered_members_isolation_and_release=True", flush=True)
    finally:
        scratch.close()
