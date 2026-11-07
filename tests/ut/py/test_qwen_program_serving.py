# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU request and lifecycle regressions; the Engine boundary never launches a worker."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_serving_effective"
spec = importlib.util.spec_from_file_location("qwen_program_serving", CASE / "run_program_serving.py")
assert spec is not None and spec.loader is not None
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


def fixture_data():
    first = torch.arange(16, dtype=torch.int32)
    outputs = torch.stack([first + 16 * (i + 1) for i in range(127)])
    fixture = SimpleNamespace(
        metadata={"prompt_token_ids": torch.tensor([[9, 8]] * 16), "first_generated_token_ids": first}
    )
    golden = {"decode_output_token_ids": outputs, "decode_input_token_ids": torch.cat([first[None], outputs[:-1]])}
    return fixture, golden


def test_reference_includes_prefill_and_every_decode_token():
    fixture, golden = fixture_data()
    expected = consumer.expected_sequences(fixture, golden, [9, 8], requests=16, max_tokens=128)
    assert expected[15] == list(range(15, 2048, 16))
    assert len(expected) == 16
    assert all(len(row) == 128 for row in expected)


@pytest.mark.parametrize("fault", ["prompt", "chain", "first"])
def test_reference_rejects_different_prompt_or_broken_feedback(fault):
    fixture, golden = fixture_data()
    if fault == "prompt":
        fixture.metadata["prompt_token_ids"][7, 1] = 99
    elif fault == "chain":
        golden["decode_input_token_ids"][100, 3] = 99
    else:
        golden["decode_input_token_ids"][0, 3] = 99
    with pytest.raises(ValueError):
        consumer.expected_sequences(fixture, golden, [9, 8], requests=16, max_tokens=128)


class Engine:
    def __init__(self, *, fault=None, stop_result=True):
        self.fault = fault
        self.stop_result = stop_result
        self.events = []
        self.active = set()

    async def start(self):
        self.events.append("start")
        self.all_started = asyncio.Event()
        if self.fault == "start":
            raise RuntimeError("start failed")

    async def stop(self):
        assert not self.active
        self.events.append("stop")
        if self.stop_result == "raise":
            raise RuntimeError("stop failed")
        return self.stop_result

    async def add_request(self, request_id, prompt, config):
        assert prompt == "prompt" and config.stream
        self.active.add(request_id)
        self.events.append(request_id)
        try:
            for index, token in enumerate([10, 20, 30]):
                if len(self.active) == self.parties:
                    self.all_started.set()
                await self.all_started.wait()
                if self.fault == "request" and index == 1:
                    raise RuntimeError("request failed")
                if self.fault == "short" and index == 2:
                    return
                yield SimpleNamespace(
                    token_id=99 if self.fault == "token" and index == 2 else token,
                    completion_tokens=index + (2 if self.fault == "count" else 1),
                    prompt_tokens=3 if self.fault == "prompt_count" else 2,
                    finished=index == 2,
                    finish_reason="error" if self.fault == "terminal" else "FINISHED_LENGTH",
                    token_ids=(10, 20, 99 if self.fault == "history" else 30) if index == 2 else (),
                )
        finally:
            self.active.remove(request_id)


def run(engine, report, requests=1, rounds=1):
    engine.parties = requests
    return asyncio.run(
        consumer.run_requests(
            engine,
            "prompt",
            SimpleNamespace(stream=True),
            [[10, 20, 30]] * requests,
            prompt_tokens=2,
            rounds=rounds,
            report=report,
        )
    )


def test_real_request_loop_reuses_engine_but_separates_requests_and_rounds():
    engine, report = Engine(), {"requests": []}
    run(engine, report, requests=16, rounds=2)
    assert engine.events[0] == "start" and engine.events[-1] == "stop"
    assert len(set(engine.events[1:-1])) == 32
    assert report["closed"] is True
    assert len(report["requests"]) == 32
    assert all(row["tokens"] == [10, 20, 30] and row["token_gate"] == "passed" for row in report["requests"])
    assert all(row["ttft_seconds"] >= 0 and row["tpot_seconds"] >= 0 for row in report["requests"])


@pytest.mark.parametrize(
    "fault", ["start", "request", "short", "token", "terminal", "history", "count", "prompt_count"]
)
def test_request_failures_stop_engine_and_preserve_partial_outputs(fault):
    engine, report = Engine(fault=fault), {"requests": []}
    with pytest.raises((RuntimeError, ValueError)):
        run(engine, report, requests=2)
    assert engine.events[-1] == "stop"
    assert report["closed"] is True
    if fault != "start":
        assert report["requests"]
        assert any(row["token_gate"] != "passed" for row in report["requests"])


@pytest.mark.parametrize("stop_result", [False, "raise"])
def test_stop_failure_never_becomes_success(stop_result):
    report = {"requests": []}
    with pytest.raises(RuntimeError):
        run(Engine(stop_result=stop_result), report)
    assert report["closed"] is False
    assert report["cleanup_error"]


def test_primary_error_survives_failed_stop():
    report = {"requests": []}
    with pytest.raises(RuntimeError, match="request failed"):
        run(Engine(fault="request", stop_result=False), report)
    assert "reaped" in report["cleanup_error"]


def test_actual_serving_engine_routes_the_request_loop_without_device_workers():
    serving = pytest.importorskip("pypto_serving.serving.engine.async_engine")
    from pypto_serving.config.types import GenerateConfig  # noqa: PLC0415

    cores = []

    class Core(Engine):
        def __init__(self, **kwargs):
            super().__init__()
            self.parties = 1
            cores.append(self)

        def pending_token_load(self):
            return 0

        async def add_request(self, request_id, prompt, config, *, on_queued, prompt_token_ids, **kwargs):
            assert prompt_token_ids == [9, 8]
            on_queued()
            async for output in super().add_request(request_id, prompt, config):
                yield serving.TokenOutput(**vars(output))

    tokenizer = SimpleNamespace(eos_token_id=None, bos_token_id=None, encode=lambda _: [9, 8])
    engine = serving.AsyncLLMEngine(
        config=serving.EngineConfig(model_id="cpu-contract", model_dir="unused"),
        tokenizer=tokenizer,
        core_factory=Core,
    )
    report = {"requests": []}
    asyncio.run(
        consumer.run_requests(
            engine,
            "prompt",
            GenerateConfig(max_new_tokens=3, stream=True),
            [[10, 20, 30]],
            prompt_tokens=2,
            rounds=2,
            report=report,
        )
    )
    assert report["closed"] is True
    assert len(report["requests"]) == 2
    assert all(row["token_gate"] == "passed" for row in report["requests"])
    assert engine._route_extra_load == [0]
    assert not engine._request_to_replica
    assert cores[0].events == ["start", "qwen-0-0", "qwen-1-0", "stop"]


def test_installed_file_is_not_misattributed_to_enclosing_worktree(tmp_path):
    path = tmp_path / "installed.py"
    path.write_text("loaded bytes")
    result = consumer.source_identity(path)
    assert "checkout_head" not in result
    assert result["sha256"] == consumer.sha256(path)


def test_cli_retains_primary_and_cleanup_failure_and_restores_environment(tmp_path, monkeypatch):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    engine = Engine(fault="request", stop_result=False)
    engine.parties = 1
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", "original")
    monkeypatch.setattr(
        consumer, "prepare", lambda *_: (engine, "prompt", SimpleNamespace(stream=True), [[10, 20, 30]], 2)
    )
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="request failed"):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--fixture",
                str(tmp_path),
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--requests",
                "1",
                "--output",
                str(output),
            ]
        )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["requests"][0]["tokens"] == [10]
    assert result["closed"] is False
    assert "request failed" in result["error"] and "reaped" in result["cleanup_error"]
    assert result["six_cell_acceptance"] == "not_qualified"
    assert os.environ["PYPTO_PROG_BUILD_DIR"] == "original"


def test_cli_retains_prepare_error_even_when_artifact_hashing_fails(tmp_path, monkeypatch):
    import json  # noqa: PLC0415

    def fail_prepare(args, report):
        artifact = args.output / "artifacts" / "partial.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"partial")
        raise ValueError("original compile error")

    def fail_hash(path):
        raise OSError("artifact unavailable")

    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    monkeypatch.setattr(consumer, "prepare", fail_prepare)
    monkeypatch.setattr(consumer, "sha256", fail_hash)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="original compile error"):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--fixture",
                str(tmp_path),
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--output",
                str(output),
            ]
        )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert "original compile error" in result["error"]
    assert "artifact unavailable" in result["artifact_error"]
    assert "PYPTO_PROG_BUILD_DIR" not in __import__("os").environ


def test_cli_refuses_existing_evidence_before_loading_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer, "prepare", lambda *_: pytest.fail("must not prepare"))
    with pytest.raises(FileExistsError):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--fixture",
                str(tmp_path),
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--output",
                str(tmp_path),
            ]
        )
