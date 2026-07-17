#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a minimal token-level O/S Perfetto trace for concurrent requests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRACE_RE = re.compile(
    r"\[STRACE\].*?\binv=(?P<inv>\d+).*?\bname=(?P<name>\S+) "
    r"ts=(?P<ts>\d+) dur=(?P<dur>\d+)(?: (?P<attrs>.*))?$"
)
TOKEN_RE = re.compile(
    r"L3_TOKEN_RECEIVED request_id=(?P<request>\d+) token_seq=(?P<seq>\d+) "
    r"received_ns=(?P<ts>\d+) final=(?P<final>True|False)"
)
DELIVERY_RE = re.compile(
    r"USER_TOKEN_DELIVERED request_id=(?P<request>\d+) token_seq=(?P<seq>\d+) "
    r"delivered_ns=(?P<ts>\d+) final=(?P<final>True|False)"
)
ORCH_SUFFIXES = ("layer_build", "epoch.materialize", "epoch.stage_upload")
TRACE_PID = 1000


def parse_attrs(raw: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in (raw or "").split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def overlap_ms(first: tuple[int, int], second: tuple[int, int]) -> float:
    return max(0, min(first[1], second[1]) - max(first[0], second[0])) / 1_000_000.0


def token_epoch(span: dict) -> int | None:
    attrs = span["attrs"]
    if span["name"].endswith("layer_build") and "layer" in attrs:
        return int(attrs["layer"]) + 1
    if "epoch" in attrs:
        return int(attrs["epoch"])
    return None


def token_orch_windows(spans: list[dict]) -> dict[int, tuple[int, int]]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for span in spans:
        if not span["name"].startswith("simpler_run.host_orch.") or not span["name"].endswith(ORCH_SUFFIXES):
            continue
        epoch = token_epoch(span)
        if epoch is not None:
            grouped.setdefault(epoch, []).append((span["ts"], span["ts"] + span["dur"]))
    return {
        epoch: (min(start for start, _end in values), max(end for _start, end in values))
        for epoch, values in grouped.items()
    }


def token_schedule_windows(spans: list[dict], tokens: list[dict]) -> dict[int, tuple[int, int]]:
    publish_ends = {
        int(span["attrs"]["epoch"]): span["ts"] + span["dur"]
        for span in spans
        if span["name"].endswith("epoch.commit") and "epoch" in span["attrs"]
    }
    receive_times = {token["seq"]: token["ts"] for token in tokens}
    return {
        epoch: (publish_end, receive_times[epoch])
        for epoch, publish_end in publish_ends.items()
        if epoch in receive_times and receive_times[epoch] >= publish_end
    }


def parse_log(log: Path) -> tuple[dict[int, list[dict]], list[dict], list[dict], list[int]]:
    spans: dict[int, list[dict]] = {}
    tokens: list[dict] = []
    deliveries: list[dict] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STRACE_RE.search(line):
            spans.setdefault(int(match.group("inv")), []).append(
                {
                    "name": match.group("name"),
                    "ts": int(match.group("ts")),
                    "dur": int(match.group("dur")),
                    "attrs": parse_attrs(match.group("attrs")),
                }
            )
        if match := TOKEN_RE.search(line):
            tokens.append(
                {
                    "request": int(match.group("request")),
                    "seq": int(match.group("seq")),
                    "ts": int(match.group("ts")),
                }
            )
        if match := DELIVERY_RE.search(line):
            deliveries.append(
                {
                    "request": int(match.group("request")),
                    "seq": int(match.group("seq")),
                    "ts": int(match.group("ts")),
                }
            )
    request_ids = list(dict.fromkeys(token["request"] for token in tokens))
    if not spans or len(request_ids) < 2:
        raise ValueError("log must contain STRACE spans and tokens for at least two requests")
    return spans, tokens, deliveries, request_ids


def map_host_invocations(spans: dict[int, list[dict]], request_ids: list[int]) -> dict[int, int]:
    normal_invs = [inv for inv, values in spans.items() if any(span["name"] == "simpler_run" for span in values)]
    prepare_invs = [
        inv for inv, values in spans.items() if any(span["name"] == "simpler_prepare_request" for span in values)
    ]
    prepare_invs.sort(
        key=lambda inv: next(span["ts"] for span in spans[inv] if span["name"] == "simpler_prepare_request")
    )
    if len(normal_invs) != 1 or len(prepare_invs) != len(request_ids) - 1:
        raise ValueError("log must contain one normal run and one Host prepare invocation per later request")
    return dict(zip(request_ids, [normal_invs[0], *prepare_invs]))


def complete(
    tid: int,
    name: str,
    category: str,
    window: tuple[int, int],
    base: int,
    *,
    request_id: int,
    token: int,
) -> dict:
    return {
        "ph": "X",
        "pid": TRACE_PID,
        "tid": tid,
        "name": name,
        "cat": category,
        "cname": "rail_animation" if category == "O" else "good",
        "ts": (window[0] - base) / 1000.0,
        "dur": (window[1] - window[0]) / 1000.0,
        "args": {"request_id": request_id, "token": token},
    }


def build_summary(
    request_ids: list[int],
    orch_by_request: dict[int, dict[int, tuple[int, int]]],
    schedule_by_request: dict[int, dict[int, tuple[int, int]]],
    tokens: list[dict],
    deliveries: list[dict],
) -> dict:
    token_parallel = {}
    for label_index, request_id in enumerate(request_ids):
        label = chr(ord("A") + label_index)
        overlaps = {
            f"O_token_{token}_vs_S_token_{token - 1}": overlap_ms(
                orch_by_request[request_id][token], schedule_by_request[request_id][token - 1]
            )
            for token in sorted(orch_by_request[request_id])
            if token - 1 in schedule_by_request[request_id]
        }
        token_parallel[f"Request_{label}"] = overlaps

    request_parallel = {}
    for previous_index, (previous_request, current_request) in enumerate(zip(request_ids, request_ids[1:])):
        previous_label = chr(ord("A") + previous_index)
        current_label = chr(ord("A") + previous_index + 1)
        for current_token, current_o in sorted(orch_by_request[current_request].items()):
            for previous_token, previous_s in sorted(schedule_by_request[previous_request].items()):
                overlap = overlap_ms(current_o, previous_s)
                if overlap > 0:
                    request_parallel[
                        f"Request_{current_label}_O_token_{current_token}_vs_Request_{previous_label}_S_token_{previous_token}"
                    ] = overlap
    received = {(token["request"], token["seq"]): token["ts"] for token in tokens}
    delivered = {(token["request"], token["seq"]): token["ts"] for token in deliveries}
    shared_keys = received.keys() & delivered.keys()
    delivery_ms = {key: (delivered[key] - received[key]) / 1_000_000.0 for key in shared_keys}
    one_to_one = received.keys() == delivered.keys() and all(delay >= 0 for delay in delivery_ms.values())
    no_batching = one_to_one and all(
        delivered[(request_id, seq)] < received[(request_id, seq + 1)]
        for request_id, seq in shared_keys
        if (request_id, seq + 1) in received
    )
    return {
        "how_to_read": "Horizontal overlap means parallel execution.",
        "token_parallel_ms": token_parallel,
        "request_parallel_ms": request_parallel,
        "stream_delivery": {
            "strict_one_to_one": one_to_one,
            "user_consumes_before_next_l3_token": no_batching,
            "l3_to_user_ms": {
                f"Request_{chr(ord('A') + label_index)}": {
                    str(seq): delivery_ms[(request_id, seq)] for req, seq in sorted(delivery_ms) if req == request_id
                }
                for label_index, request_id in enumerate(request_ids)
            },
            "max_l3_to_user_ms": max(delivery_ms.values(), default=None),
        },
        "schedule_window": "Host publish completion to L3 token receipt (inferred).",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spans, tokens, deliveries, request_ids = parse_log(args.log)
    host_inv_by_request = map_host_invocations(spans, request_ids)
    tokens_by_request = {
        request_id: [token for token in tokens if token["request"] == request_id] for request_id in request_ids
    }
    orch_by_request = {
        request_id: token_orch_windows(spans[host_inv_by_request[request_id]]) for request_id in request_ids
    }
    schedule_by_request = {
        request_id: token_schedule_windows(spans[host_inv_by_request[request_id]], tokens_by_request[request_id])
        for request_id in request_ids
    }
    all_windows = [
        window
        for request_id in request_ids
        for windows in (orch_by_request[request_id], schedule_by_request[request_id])
        for window in windows.values()
    ]
    if not all_windows:
        raise ValueError("log contains no token-level O/S windows")
    base = min(start for start, _end in all_windows)

    events = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": TRACE_PID,
            "tid": 0,
            "args": {"name": "O/S Pipeline + per-token stream"},
        }
    ]
    deliveries_by_key = {(delivery["request"], delivery["seq"]): delivery for delivery in deliveries}
    for label_index, request_id in enumerate(request_ids):
        label = chr(ord("A") + label_index)
        orch_tid = label_index * 3 + 10
        schedule_tid = orch_tid + 1
        stream_tid = orch_tid + 2
        events.extend(
            [
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": TRACE_PID,
                    "tid": orch_tid,
                    "args": {"name": f"Request {label} · O (Orch)"},
                },
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": TRACE_PID,
                    "tid": schedule_tid,
                    "args": {"name": f"Request {label} · S (Schedule)"},
                },
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": TRACE_PID,
                    "tid": stream_tid,
                    "args": {"name": f"Request {label} · Stream (L3 → User)"},
                },
            ]
        )
        for token, window in sorted(orch_by_request[request_id].items()):
            events.append(complete(orch_tid, f"O token {token}", "O", window, base, request_id=request_id, token=token))
        for token, window in sorted(schedule_by_request[request_id].items()):
            events.append(
                complete(schedule_tid, f"S token {token}", "S", window, base, request_id=request_id, token=token)
            )
        for receipt in tokens_by_request[request_id]:
            delivery = deliveries_by_key.get((request_id, receipt["seq"]))
            if delivery is None:
                continue
            flow_id = request_id * 100 + receipt["seq"]
            delay_ms = (delivery["ts"] - receipt["ts"]) / 1_000_000.0
            events.extend(
                [
                    {
                        "ph": "s",
                        "pid": TRACE_PID,
                        "tid": schedule_tid,
                        "name": f"L3 token {receipt['seq']}",
                        "cat": "stream_delivery",
                        "id": flow_id,
                        "ts": (receipt["ts"] - base) / 1000.0,
                    },
                    {
                        "ph": "f",
                        "bp": "e",
                        "pid": TRACE_PID,
                        "tid": stream_tid,
                        "name": f"User token {receipt['seq']}",
                        "cat": "stream_delivery",
                        "id": flow_id,
                        "ts": (delivery["ts"] - base) / 1000.0,
                    },
                    {
                        "ph": "i",
                        "s": "t",
                        "pid": TRACE_PID,
                        "tid": stream_tid,
                        "name": f"User token {receipt['seq']}",
                        "cat": "stream_delivery",
                        "cname": "rail_response",
                        "ts": (delivery["ts"] - base) / 1000.0,
                        "args": {"request_id": request_id, "token": receipt["seq"], "l3_to_user_ms": delay_ms},
                    },
                ]
            )

    summary = build_summary(request_ids, orch_by_request, schedule_by_request, tokens, deliveries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"traceEvents": events, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
