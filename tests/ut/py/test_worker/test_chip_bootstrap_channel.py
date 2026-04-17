# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sys
import traceback

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
for _binding_dir in sorted((ROOT / "build").glob("*/python/bindings")):
    sys.path.insert(0, str(_binding_dir))

from simpler_setup.code_runner import CodeRunner, create_code_runner  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ChipCallConfig,
    ChipStorageTaskArgs,
    DataType,
    TaskArgs,
    TensorArgType,
    make_tensor_arg,
)
from simpler.worker import (  # noqa: E402
    ChipBootstrapConfig,
    ChipBootstrapReply,
    ChipBufferSpec,
    ChipCommBootstrapConfig,
    ChipContext,
    HostBufferStaging,
    Worker,
)
from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: E402
from tests.st.a2a3.tensormap_and_ringbuffer.test_l3_group import TestL3Group as _L3GroupScene  # noqa: E402


def _create_shm_with_bytes(data: bytes) -> tuple[SharedMemory, HostBufferStaging]:
    shm = SharedMemory(create=True, size=len(data))
    assert shm.buf is not None
    if data:
        shm.buf[: len(data)] = data
    return shm, HostBufferStaging(name="", shm_name=shm.name, size=len(data))


def _create_empty_shm(size: int) -> tuple[SharedMemory, HostBufferStaging]:
    shm = SharedMemory(create=True, size=size)
    return shm, HostBufferStaging(name="", shm_name=shm.name, size=size)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def _cleanup_shms(shms: list[SharedMemory]) -> None:
    for shm in shms:
        try:
            shm.close()
        finally:
            shm.unlink()


def _bootstrap_rank_entry(
    host_path: str,
    aicpu_path: str,
    aicore_path: str,
    sim_context_path: str,
    device_id: int,
    cfg: ChipBootstrapConfig,
    expected_bytes: bytes,
    result_queue,
) -> None:
    from simpler.task_interface import ChipWorker  # noqa: PLC0415

    worker = ChipWorker()
    try:
        worker.init(host_path, aicpu_path, aicore_path, sim_context_path)
        reply = worker.bootstrap_context(device_id, cfg)
        copied = worker.copy_device_to_bytes(reply.buffer_ptrs[0], len(expected_bytes))
        worker.comm_barrier(reply.comm_handle)
        worker.shutdown_bootstrap_context(cfg, comm_handle=reply.comm_handle, buffer_ptrs=reply.buffer_ptrs)
        result_queue.put(
            {
                "ok": True,
                "reply": {
                    "comm_handle": reply.comm_handle,
                    "device_ctx": reply.device_ctx,
                    "local_window_base": reply.local_window_base,
                    "actual_window_size": reply.actual_window_size,
                    "buffer_ptrs": list(reply.buffer_ptrs),
                },
                "copied": copied,
            }
        )
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})
        raise
    finally:
        worker.finalize()


def _level3_bootstrap_entry(
    device_ids: list[int],
    bootstrap_configs: list[ChipBootstrapConfig],
    result_queue,
) -> None:
    try:
        callable_obj = _L3GroupScene().build_callable("a2a3")
        vector_kernel = callable_obj["vector_kernel"]

        size = 128 * 128
        config = ChipCallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        host_inputs_a = [
            torch.full((size,), 2.0, dtype=torch.float32).share_memory_(),
            torch.full((size,), 4.0, dtype=torch.float32).share_memory_(),
        ]
        host_inputs_b = [
            torch.full((size,), 3.0, dtype=torch.float32).share_memory_(),
            torch.full((size,), 5.0, dtype=torch.float32).share_memory_(),
        ]
        host_outputs_f = [
            torch.zeros(size, dtype=torch.float32).share_memory_(),
            torch.zeros(size, dtype=torch.float32).share_memory_(),
        ]

        observed: dict[str, object] = {}

        def orch_fn(o, _args, _cfg):
            contexts = sorted(worker.chip_contexts, key=lambda ctx: ctx.rank)
            observed["contexts"] = [
                {
                    "rank": ctx.rank,
                    "nranks": ctx.nranks,
                    "device_id": ctx.device_id,
                    "device_ctx": ctx.device_ctx,
                    "buffer_ptrs": dict(ctx.buffer_ptrs),
                }
                for ctx in contexts
            ]

            args_list = []
            for rank, _ctx in enumerate(contexts):
                task_args = TaskArgs()
                task_args.add_tensor(make_tensor_arg(host_inputs_a[rank]), TensorArgType.INPUT)
                task_args.add_tensor(make_tensor_arg(host_inputs_b[rank]), TensorArgType.INPUT)
                task_args.add_tensor(make_tensor_arg(host_outputs_f[rank]), TensorArgType.OUTPUT_EXISTING)
                args_list.append(task_args)

            o.submit_next_level_group(vector_kernel, args_list, config)

        worker = Worker(
            level=3,
            device_ids=device_ids,
            num_sub_workers=0,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            build=True,
            chip_bootstrap_configs=bootstrap_configs,
        )

        try:
            worker.init()
            worker.run(orch_fn)

            expected_rank0 = (host_inputs_a[0] + host_inputs_b[0] + 1) * (host_inputs_a[0] + host_inputs_b[0] + 2) + (
                host_inputs_a[0] + host_inputs_b[0]
            )
            expected_rank1 = (host_inputs_a[1] + host_inputs_b[1] + 1) * (host_inputs_a[1] + host_inputs_b[1] + 2) + (
                host_inputs_a[1] + host_inputs_b[1]
            )

            if not torch.allclose(host_outputs_f[0], expected_rank0):
                raise AssertionError("rank 0 output mismatch")
            if not torch.allclose(host_outputs_f[1], expected_rank1):
                raise AssertionError("rank 1 output mismatch")

            result_queue.put(
                {
                    "ok": True,
                    "contexts": observed["contexts"],
                    "output_values": [float(host_outputs_f[0][0]), float(host_outputs_f[1][0])],
                }
            )
        finally:
            worker.close()
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})
        raise


def _make_bootstrap_config(
    *,
    rank: int,
    nranks: int,
    rootinfo_path: Path,
    window_size: int,
    buffer_specs: list[ChipBufferSpec],
    host_inputs: list[HostBufferStaging],
    host_outputs: list[HostBufferStaging] | None = None,
) -> ChipBootstrapConfig:
    named_inputs = []
    for buf, staging in zip([b for b in buffer_specs if b.load_from_host], host_inputs, strict=True):
        named_inputs.append(HostBufferStaging(name=buf.name, shm_name=staging.shm_name, size=staging.size))

    named_outputs = []
    if host_outputs is not None:
        for buf, staging in zip([b for b in buffer_specs if b.store_to_host], host_outputs, strict=True):
            named_outputs.append(HostBufferStaging(name=buf.name, shm_name=staging.shm_name, size=staging.size))

    return ChipBootstrapConfig(
        comm=ChipCommBootstrapConfig(
            rank=rank,
            nranks=nranks,
            rootinfo_path=str(rootinfo_path),
            window_size=window_size,
        ),
        buffers=buffer_specs,
        host_inputs=named_inputs,
        host_outputs=named_outputs,
    )


class TestDistributedWorkerApi:
    def test_chip_context_exposes_buffer_tensors(self):
        cfg = ChipBootstrapConfig(
            buffers=[ChipBufferSpec(name="buf", dtype="float32", count=4, placement="window", nbytes=16)]
        )
        ctx = ChipContext(
            bootstrap_config=cfg,
            device_id=2,
            bootstrap_reply=ChipBootstrapReply(
                device_ctx=10,
                local_window_base=20,
                actual_window_size=32,
                buffer_ptrs=[0x9000],
            ),
            buffer_tensors={"buf": cfg.buffers[0].make_tensor_arg(0x9000)},
        )

        tensor = ctx.buffer_tensors["buf"]
        assert tensor.data == 0x9000
        assert tensor.dtype == DataType.FLOAT32
        assert tensor.shapes == (4,)

    # Minimal true correctness baseline: real a2a3 runtime, one L2 worker,
    # one real callable, and a direct golden check on the host-visible output.
    @pytest.mark.platforms(["a2a3"])
    @pytest.mark.device_count(1)
    def test_single_chip_runtime_end_to_end_correctness(self, st_device_ids, request):
        callable_obj = _L3GroupScene().build_callable("a2a3")
        vector_kernel = callable_obj["vector_kernel"]

        size = 128 * 128
        a = torch.full((size,), 2.0, dtype=torch.float32)
        b = torch.full((size,), 3.0, dtype=torch.float32)
        f = torch.zeros(size, dtype=torch.float32)

        args = ChipStorageTaskArgs()
        args.add_tensor(make_tensor_arg(a))
        args.add_tensor(make_tensor_arg(b))
        args.add_tensor(make_tensor_arg(f))

        config = ChipCallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        worker = Worker(
            level=2,
            device_id=st_device_ids[0],
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            build=request.config.getoption("--build", default=False),
        )

        try:
            worker.init()
            worker.run(vector_kernel, args, config=config)
            expected = (a + b + 1) * (a + b + 2) + (a + b)
            assert torch.allclose(f, expected)
        finally:
            worker.close()

    # L3 integration smoke: verifies bootstrap-enabled L3 startup, real child
    # processes, real callable compilation, and real group submit. It does not
    # assert full L3->L2->L1 golden correctness on hardware yet.
    @pytest.mark.platforms(["a2a3"])
    @pytest.mark.device_count(2)
    def test_level3_bootstrap_runs_real_runtime_kernel(self, st_device_ids, tmp_path):
        bootstrap_payloads = [b"BOOT0", b"BOOT1"]
        input_shms: list[SharedMemory] = []
        bootstrap_configs = []
        for rank, payload in enumerate(bootstrap_payloads):
            payload_shm, payload_stage = _create_shm_with_bytes(payload)
            input_shms.append(payload_shm)
            bootstrap_configs.append(
                _make_bootstrap_config(
                    rank=rank,
                    nranks=2,
                    rootinfo_path=tmp_path / "worker_rootinfo.bin",
                    window_size=4096,
                    buffer_specs=[
                        ChipBufferSpec(
                            name="payload",
                            dtype="uint8",
                            count=len(payload),
                            placement="window",
                            nbytes=len(payload),
                            load_from_host=True,
                        ),
                    ],
                    host_inputs=[payload_stage],
                )
            )

        ctx = mp.get_context("fork")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_level3_bootstrap_entry,
            args=(st_device_ids, bootstrap_configs, result_queue),
            name="level3-bootstrap-e2e",
        )
        try:
            proc.start()
            result = result_queue.get(timeout=180)
            proc.join(timeout=180)
            assert proc.exitcode == 0
            assert result["ok"], result

            contexts = result["contexts"]
            assert contexts is not None
            assert [ctx["rank"] for ctx in contexts] == [0, 1]
            assert [ctx["nranks"] for ctx in contexts] == [2, 2]
            assert [ctx["device_id"] for ctx in contexts] == st_device_ids
            for ctx in contexts:
                assert ctx["device_ctx"] != 0
                assert set(ctx["buffer_ptrs"]) == {"payload"}
            assert result["output_values"] == [47.0, 119.0]
        finally:
            result_queue.close()
            _cleanup_shms(input_shms)

    # Bootstrap-only correctness: exercises the real onboard init/comm/window
    # path and proves staged host bytes reach device memory on both ranks.
    @pytest.mark.platforms(["a2a3"])
    @pytest.mark.device_count(2)
    def test_bootstrap_context_uses_real_a2a3_runtime(self, st_device_ids, tmp_path, request):
        binaries = RuntimeBuilder("a2a3").get_binaries(
            "tensormap_and_ringbuffer",
            build=request.config.getoption("--build", default=False),
        )
        host_path = str(binaries.host_path)
        aicpu_path = str(binaries.aicpu_path)
        aicore_path = str(binaries.aicore_path)
        sim_context_path = str(binaries.sim_context_path) if binaries.sim_context_path else ""

        input_payloads = [b"RANK0", b"RANK1"]
        input_shms = []
        configs = []
        for rank, payload in enumerate(input_payloads):
            shm, staging = _create_shm_with_bytes(payload)
            input_shms.append(shm)
            cfg = _make_bootstrap_config(
                rank=rank,
                nranks=2,
                rootinfo_path=tmp_path / "bootstrap_rootinfo.bin",
                window_size=4096,
                buffer_specs=[
                    ChipBufferSpec(
                        name="payload",
                        dtype="uint8",
                        count=len(payload),
                        placement="window",
                        nbytes=len(payload),
                        load_from_host=True,
                    )
                ],
                host_inputs=[staging],
            )
            configs.append(cfg)

        ctx = mp.get_context("fork")
        result_queue = ctx.Queue()
        procs = []
        try:
            for rank, (device_id, cfg, payload) in enumerate(zip(st_device_ids, configs, input_payloads, strict=True)):
                proc = ctx.Process(
                    target=_bootstrap_rank_entry,
                    args=(host_path, aicpu_path, aicore_path, sim_context_path, device_id, cfg, payload, result_queue),
                    name=f"bootstrap-rank-{rank}",
                )
                proc.start()
                procs.append(proc)

            results = [result_queue.get(timeout=180) for _ in range(2)]
            for proc in procs:
                proc.join(timeout=180)
                assert proc.exitcode == 0

            assert all(result["ok"] for result in results), results
            copied_payloads = {result["copied"] for result in results}
            assert copied_payloads == set(input_payloads)
            for result in results:
                reply = result["reply"]
                assert reply["comm_handle"] != 0
                assert reply["device_ctx"] != 0
                assert reply["local_window_base"] != 0
                assert reply["actual_window_size"] >= 4096
                assert len(reply["buffer_ptrs"]) == 1
        finally:
            _cleanup_shms(input_shms)
            result_queue.close()

    def test_create_code_runner_returns_unified_code_runner_for_distributed(self):
        # create_code_runner still expects the legacy "kernel_config.py +
        # golden.py" layout. The a2a3 examples in this repo use scene-test
        # modules instead, so keep this smoke test on an a5 example that
        # matches the current input contract.
        kernels_dir = ROOT / "examples" / "a5" / "host_build_graph" / "paged_attention" / "kernels"
        golden_path = ROOT / "examples" / "a5" / "host_build_graph" / "paged_attention" / "golden.py"

        old_platform = os.environ.get("PTO_PLATFORM")
        os.environ["PTO_PLATFORM"] = "a5"
        try:
            runner = create_code_runner(
                kernels_dir=str(kernels_dir),
                golden_path=str(golden_path),
                platform="a5",
                nranks=2,
                device_ids=[0, 1],
            )
        finally:
            if old_platform is None:
                os.environ.pop("PTO_PLATFORM", None)
            else:
                os.environ["PTO_PLATFORM"] = old_platform

        assert isinstance(runner, CodeRunner)
        assert runner._is_distributed is True
        assert runner.nranks == 2
        assert runner.device_ids == [0, 1]
