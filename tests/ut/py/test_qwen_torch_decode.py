# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-request ABI, page isolation and borrowed-resource lifetime controls."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))
    spec = importlib.util.spec_from_file_location("qwen_torch_decode_test", CASE / "torch_decode.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_cache_layout_and_capacity(adapter, monkeypatch):
    for name, value in {"LAYERS": 2, "BATCH": 2, "PAGE": 2, "HEADS": 2, "HEAD_DIM": 4}.items():
        monkeypatch.setattr(adapter, name, value)
    key, value = adapter.allocate_layer_caches(pages=56, device="cpu")
    assert key.shape == value.shape == (2, 56, 2, 2, 4)
    key[1, 0, 0, 0, 0] = 7
    assert key.is_contiguous() and key.dtype == torch.bfloat16
    assert torch.count_nonzero(key[0]) == torch.count_nonzero(value) == 0
    with pytest.raises(ValueError, match="56 pages"):
        adapter.allocate_layer_caches(pages=55, device="cpu")


@pytest.mark.parametrize("names,signature", [((), [object()]), (("input",), [])])
def test_argument_count_mismatch_refused_before_compile(adapter, names, signature):
    obj = adapter.TorchDecode.__new__(adapter.TorchDecode)
    obj.artifact = SimpleNamespace(
        parameter_names=names,
        callable_spec={"orchestration": {"signature": signature}},
        compile=lambda: pytest.fail("invalid argument counts reached compilation"),
    )
    with pytest.raises(ValueError, match="parameter and signature counts"):
        obj._initialize()


@pytest.fixture
def backend(adapter, monkeypatch):
    calls = []
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: calls.append("framework-drained"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda _: stream), raising=False)

    class Run:
        def wait(self, timeout):
            assert timeout == 120
            calls.append("retired")

    class Worker:
        def submit(self, *args, caller_stream):
            assert caller_stream == 123
            calls.append("submitted")
            return Run()

        def close(self):
            calls.append("worker-closed")

    obj = adapter.TorchDecode.__new__(adapter.TorchDecode)
    obj.failed, obj.closed, obj.steps = False, False, 0
    obj.position, obj.blocks, obj.padding_pages, obj.request_id = None, None, None, None
    obj.caches = [torch.empty((40, 448), device="meta")]
    obj.device, obj.worker, obj.run = torch.device("cpu"), Worker(), None
    obj.callable, obj.args, obj.config = object(), object(), object()
    obj.sources = [SimpleNamespace(close=lambda: calls.append("source-closed"))]
    obj.values = {
        "sampled_ids_in": torch.zeros((16, 8), dtype=torch.int32),
        "seq_lens": torch.zeros(16, dtype=torch.int32),
        "block_table": torch.zeros(16 * 28, dtype=torch.int32),
        "slot_mapping": torch.zeros(16, dtype=torch.int32),
        "out": torch.arange(16 * 7, dtype=torch.float32).reshape(16, 7),
    }
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([3339], dtype=torch.int32),
        block_tables=torch.arange(28, dtype=torch.int32).reshape(1, 28),
        slot_mapping=torch.tensor([3338], dtype=torch.int32),
    )
    return obj, metadata, calls, stream


def test_token_feedback_ordering_and_close(backend):
    obj, metadata, calls, _ = backend
    result = obj(torch.tensor([32313]), torch.tensor([3338]), metadata, ("request",))
    assert result.shape == (1, 7)
    assert calls == ["framework-drained", "submitted", "retired"]
    assert obj.values["sampled_ids_in"][:, 0].tolist() == [32313] * 16
    metadata.seq_lens += 1
    metadata.slot_mapping += 1
    sampled = result.argmax(-1)
    obj(sampled, torch.tensor([3339]), metadata, ("request",))
    assert obj.values["sampled_ids_in"][:, 0].tolist() == [int(sampled[0])] * 16
    assert obj.steps == 2
    obj.close()
    assert calls[-4:] == ["retired", "framework-drained", "worker-closed", "source-closed"]
    calls.clear()
    obj.close()
    assert calls == []
    with pytest.raises(RuntimeError, match="failed decode"):
        obj(sampled, torch.tensor([3340]), metadata, ("request",))


@pytest.mark.parametrize("columns", [27, 28, 32])
def test_padding_uses_private_pages_in_bound_table(backend, columns):
    obj, metadata, calls, _ = backend
    metadata.block_tables = torch.full((1, columns), -1, dtype=torch.int32)
    metadata.block_tables[0, :27] = torch.arange(27, dtype=torch.int32)
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    table = obj.values["block_table"].view(16, 28)
    padding = table[1:].reshape(-1).tolist()
    assert len(set(padding)) == 15 * 28
    assert set(table[0, :27].tolist()).isdisjoint(padding)
    assert calls == ["framework-drained", "submitted", "retired"]


def test_page_growth_reselects_conflicting_padding(backend):
    obj, metadata, _, _ = backend
    metadata.block_tables[0, 27] = -1
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    next_page = int(obj.values["block_table"].view(16, 28)[1, 0])
    metadata.block_tables[0, 27] = next_page
    metadata.seq_lens.fill_(3457)
    metadata.slot_mapping.fill_(next_page * 128)
    obj.position = 3455
    obj(torch.ones(1), torch.tensor([3456]), metadata, ("request",))
    assert next_page not in obj.values["block_table"].view(16, 28)[1:].reshape(-1).tolist()


def test_padding_avoids_all_reserved_pages(backend):
    obj, metadata, _, _ = backend
    metadata.block_tables[0, 27] = 421
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    assert 421 not in obj.values["block_table"].view(16, 28)[1:].reshape(-1).tolist()


def test_populated_extra_block_columns_are_rejected_before_submit(backend):
    obj, metadata, calls, _ = backend
    metadata.block_tables = torch.arange(32, dtype=torch.int32).reshape(1, -1)
    with pytest.raises(ValueError, match="beyond"):
        obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    assert calls == [] and not obj.failed


def test_timeout_retains_backing_and_close_waits_before_framework_sync(backend):
    obj, metadata, calls, _ = backend

    def wait(timeout):
        assert timeout == 120
        raise TimeoutError("completion unknown")

    obj.worker.submit = lambda *args, **kwargs: SimpleNamespace(wait=wait)
    with pytest.raises(TimeoutError, match="step=0.*position=3338"):
        obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    assert obj.failed and obj.steps == 0
    calls.clear()
    with pytest.raises(TimeoutError, match="close.*step=0"):
        obj.close()
    assert calls == [] and obj.sources and not obj.closed
    obj.run.wait = lambda timeout: calls.append("retired")
    obj.close()
    assert obj.closed and calls == ["retired", "framework-drained", "worker-closed", "source-closed"]


@pytest.mark.parametrize(
    "change", ["identity", "position", "batch", "table", "range", "shared", "moved", "length", "slot"]
)
def test_invalid_input_refused_before_submission_and_recoverable(backend, change):
    obj, metadata, calls, _ = backend
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    calls.clear()
    metadata.seq_lens += 1
    metadata.slot_mapping += 1
    positions, ids, identities = torch.tensor([3339]), torch.ones(1), ("request",)
    original_table = metadata.block_tables.clone()
    if change == "identity":
        identities = ("replacement",)
    elif change == "position":
        positions += 1
    elif change == "batch":
        ids = torch.ones(16)
    elif change == "table":
        metadata.block_tables = metadata.block_tables[:, :26]
    elif change == "range":
        positions.fill_(3465)
    elif change == "shared":
        metadata.block_tables[0, 0] = metadata.block_tables[0, 1]
    elif change == "moved":
        metadata.block_tables[0, 0] = 447
    elif change == "length":
        metadata.seq_lens -= 1
    elif change == "slot":
        metadata.slot_mapping += 1
    with pytest.raises(ValueError):
        obj(ids, positions, metadata, identities)
    assert not obj.failed and calls == [] and obj.steps == 1
    metadata.block_tables = original_table
    metadata.seq_lens.fill_(3340)
    metadata.slot_mapping.fill_(3339)
    obj(torch.ones(1), torch.tensor([3339]), metadata, ("request",))
    assert obj.steps == 2


@pytest.mark.parametrize("stage", ["metadata", "sync", "submit", "wait"])
def test_unknown_failure_latches_and_preserves_sources(backend, monkeypatch, stage):
    obj, metadata, calls, stream = backend

    def fail(*args, **kwargs):
        raise RuntimeError("completion unproven")

    if stage == "metadata":
        monkeypatch.setattr(obj, "_prepare_single_request", fail)
    elif stage == "sync":
        stream.synchronize = fail
    elif stage == "submit":
        obj.worker.submit = fail
    else:
        obj.worker.submit = lambda *args, **kwargs: SimpleNamespace(wait=fail)
    with pytest.raises(RuntimeError, match="unproven"):
        obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    assert obj.failed
    with pytest.raises(RuntimeError, match="failed decode"):
        obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    assert "source-closed" not in calls


def test_close_failure_retains_sources_for_retry(backend):
    obj, metadata, calls, _ = backend
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))

    def fail():
        raise RuntimeError("worker not drained")

    obj.worker.close = fail
    with pytest.raises(RuntimeError, match="not drained"):
        obj.close()
    assert obj.failed and not obj.closed and "source-closed" in calls
    obj.worker.close = lambda: calls.append("worker-closed")
    obj.close()
    assert obj.closed and calls[-1] == "worker-closed" and calls.count("source-closed") == 1


def test_terminal_run_error_still_cleans_up_and_does_not_rewait(backend):
    obj, metadata, calls, _ = backend
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    primary = RuntimeError("device execution failed")
    obj.run.wait = lambda timeout: (_ for _ in ()).throw(primary)
    with pytest.raises(RuntimeError) as caught:
        obj.close()
    assert caught.value is primary
    assert calls[-2:] == ["worker-closed", "source-closed"]
    assert obj.closed and obj.run is None and obj.worker is None and not obj.sources
    obj.close()
    assert calls.count("source-closed") == 1


@pytest.mark.parametrize("target", ["stream", "worker", "source"])
def test_terminal_run_and_cleanup_errors_remain_distinct(backend, monkeypatch, target):
    obj, metadata, calls, stream = backend
    obj(torch.ones(1), torch.tensor([3338]), metadata, ("request",))
    primary = RuntimeError("device execution failed")
    cleanup = OSError("cleanup failed")
    obj.run.wait = lambda timeout: (_ for _ in ()).throw(primary)
    owner, method = {
        "stream": (stream, "synchronize"),
        "worker": (obj.worker, "close"),
        "source": (obj.sources[0], "close"),
    }[target]
    original = getattr(owner, method)
    monkeypatch.setattr(owner, method, lambda: (_ for _ in ()).throw(cleanup))
    with pytest.raises(RuntimeError) as caught:
        obj.close()
    assert caught.value is primary and primary.__context__ is cleanup
    assert obj.cleanup_error is cleanup and obj.failed and not obj.closed
    assert obj.run is None
    assert (obj.worker is not None) == (target == "worker")
    assert bool(obj.sources) == (target == "source")
    monkeypatch.setattr(owner, method, original)
    obj.close()
    assert obj.closed and obj.worker is None and not obj.sources
