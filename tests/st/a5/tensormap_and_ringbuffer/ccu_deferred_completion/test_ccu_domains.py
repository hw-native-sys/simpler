# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real CCU domain reuse, retirement and hidden callable isolation on two A5 ranks."""

import os
import shlex
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CcuFunction, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]
_RUNTIME = "tensormap_and_ringbuffer"
_ELEMS = 2048
_RESULT_ROUNDS = 16


def _compile_root(platform, directory, root):
    module = directory / f"ccu_domains_root{root}.so"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(_ROOT / "src/common"),
            str(_HERE / "kernels/host/ccu_positional.cpp"),
            "-o",
            str(module),
        ],
        check=True,
    )
    binary = module.read_bytes()
    module.unlink()
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/ccu_domains_orch.cpp"),
            "function_name": "ccu_domains_orchestration",
            "config_name": "ccu_domains_orchestration_config",
            "signature": [D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": index,
                "source": str(_HERE / f"kernels/aiv/{name}.cpp"),
                "core_type": "aiv",
                "signature": signature,
            }
            for index, name, signature in [
                (0, "initialize", [D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR]),
                (1, "domains_submit", [D.IN, D.INOUT, D.IN, D.INOUT, D.SCALAR, D.SCALAR]),
                (2, "domains_verify", [D.IN, D.IN, D.IN, D.INOUT, D.SCALAR, D.SCALAR]),
            ]
        ],
        # Each callable deliberately has its own ID 0, with different broadcast semantics.
        "ccu_functions": [CcuFunction.build(binary, f"broadcast_root{root}")],
    }
    key = l3_compile_cache_key(__name__, "CcuDomains", f"root{root}", platform, _RUNTIME)
    return compile_chip_callable_spec(spec, platform, _RUNTIME, key)


def _domain(orch, name, registered):
    return orch.get_or_allocate_domain(
        name=name,
        members=[0, 1],
        workspace_engines=("ccu",),
        ccu_callables=(registered,),
        window_size=4 * 1024 * 1024,
        buffers=[CommBufferSpec(name, "float32", _ELEMS, _ELEMS * 4) for name in ("source", "destination", "scratch")],
        retain_after_run=True,
    )


def _results(worker):
    outputs = [worker.create_buffer(_RESULT_ROUNDS * 16) for _ in range(2)]
    for output in outputs:
        output.shm.buf[:] = bytes([0xFF]) * (_RESULT_ROUNDS * 16)
    return outputs


def _submit(orch, cfg, registered, domain, outputs, seed, *, verify_only=False):
    assert 0 <= seed < _RESULT_ROUNDS
    for rank in range(2):
        args = TaskArgs()
        for name in ("source", "destination", "scratch"):
            args.add_tensor(
                domain.local_context(rank).buffers[name].tensor((_ELEMS,), DataType.FLOAT32), TensorArgType.INOUT
            )
        args.add_tensor(outputs[rank].tensor((_RESULT_ROUNDS * 4,), DataType.INT32), TensorArgType.INOUT)
        args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
        args.add_scalar(seed)
        args.add_scalar(int(verify_only))
        orch.submit_next_level(registered, args, cfg, worker=rank)


def _check(outputs, seed):
    expected = (0, (seed // 2) % 2, 1024 if seed % 2 else 64, 16 + seed % 8 * 16)
    for rank, output in enumerate(outputs):
        actual = struct.unpack_from("<4i", output.shm.buf, seed * 16)
        assert actual == expected, f"rank={rank}, seed={seed}: {actual} != {expected}"


def _new_worker(platform, devices):
    return Worker(
        level=3,
        platform=platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in devices[:2]],
        num_sub_workers=0,
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_ccu_retained_domains_reuse_release_and_recreate(st_platform, st_device_ids, tmp_path):
    worker = _new_worker(st_platform, st_device_ids)
    registered = worker.register(_compile_root(st_platform, tmp_path, 0))
    saved = {}
    try:
        worker.init()
        outputs = {name: _results(worker) for name in ("domain_a", "domain_b")}

        def first(orch, _args, cfg):
            # All communication preparation precedes the first device submission.
            saved.update({name: _domain(orch, name, registered) for name in outputs})
            for name, seed in (("domain_a", 0), ("domain_b", 4)):
                _submit(orch, cfg, registered, saved[name], outputs[name], seed)

        worker.run(first)
        old_a, domain_b = saved["domain_a"], saved["domain_b"]
        old_ref, b_ref = old_a.ref, domain_b.ref
        addresses = {name: tuple(domain.device_context(rank) for rank in range(2)) for name, domain in saved.items()}
        assert old_ref != b_ref
        assert all(a != b for a, b in zip(addresses["domain_a"], addresses["domain_b"], strict=True))
        assert not old_a.released and not domain_b.released
        _check(outputs["domain_a"], 0)
        _check(outputs["domain_b"], 4)

        def reuse_a(orch, _args, cfg):
            domain = _domain(orch, "domain_a", registered)
            assert domain is old_a and domain.ref == old_ref
            assert tuple(domain.device_context(rank) for rank in range(2)) == addresses["domain_a"]
            _submit(orch, cfg, registered, domain, outputs["domain_a"], 1)

        worker.run(reuse_a)
        _check(outputs["domain_a"], 1)
        old_a.release()
        assert old_a.released and old_a.freed
        assert set(worker.live_domains) == {"domain_b"}
        with pytest.raises(RuntimeError, match="released"):
            old_a.device_context(0, workspace_engines=("ccu",))

        def verify_survivor(orch, _args, cfg):
            with pytest.raises(KeyError, match="no live grant"):
                orch.get_domain(old_ref)
            domain = _domain(orch, "domain_b", registered)
            assert domain is domain_b and domain.ref == b_ref
            assert tuple(domain.device_context(rank) for rank in range(2)) == addresses["domain_b"]
            _submit(orch, cfg, registered, domain, outputs["domain_b"], 4, verify_only=True)

        worker.run(verify_survivor)
        _check(outputs["domain_b"], 4)

        def survivor_and_replacement(orch, _args, cfg):
            replacement = _domain(orch, "domain_a", registered)
            survivor = _domain(orch, "domain_b", registered)
            assert replacement is not old_a and replacement.ref != old_ref
            assert replacement.ref.owner_instance_id == old_ref.owner_instance_id
            assert replacement.generation > 0
            assert (replacement.domain_id, replacement.generation) != (old_a.domain_id, old_a.generation)
            assert survivor is domain_b and survivor.ref == b_ref
            assert orch.get_domain(replacement.ref) is replacement
            with pytest.raises(RuntimeError, match="released"):
                old_a.device_context(0, workspace_engines=("ccu",))
            with pytest.raises(KeyError, match="no live grant"):
                orch.get_domain(old_ref)
            saved["replacement"] = replacement
            _submit(orch, cfg, registered, survivor, outputs["domain_b"], 5)
            _submit(orch, cfg, registered, replacement, outputs["domain_a"], 8)

        worker.run(survivor_and_replacement)
        _check(outputs["domain_b"], 5)
        _check(outputs["domain_a"], 8)
        for domain in (saved["replacement"], domain_b):
            domain.release()
            assert domain.freed
        assert worker.live_domains == {}
        worker.unregister(registered)
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_ccu_id_zero_isolated_after_new_callable_directory_publication(st_platform, st_device_ids, tmp_path):
    root0 = _compile_root(st_platform, tmp_path, 0)
    root1 = _compile_root(st_platform, tmp_path, 1)
    worker = _new_worker(st_platform, st_device_ids)
    first = worker.register(root0)
    saved = []
    identities = []
    try:
        worker.init()
        outputs = _results(worker)

        def run(registered, seed):
            def orchestration(orch, _args, cfg):
                domain = _domain(orch, "shared_domain", registered)
                if saved:
                    assert domain is saved[0] and domain.ref == identities[0]
                else:
                    saved.append(domain)
                    identities.append(domain.ref)
                _submit(orch, cfg, registered, domain, outputs, seed)

            worker.run(orchestration)
            _check(outputs, seed)

        run(first, 0)
        # Late registration forces domain reuse to publish a new two-instance directory.
        second = worker.register(root1)
        assert first.digest != second.digest
        run(second, 2)
        run(first, 1)
        run(second, 3)
        saved[0].release()
        assert saved[0].freed and worker.live_domains == {}
        worker.unregister(first)
        worker.unregister(second)
    finally:
        worker.close()
