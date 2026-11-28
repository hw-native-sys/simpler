# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pair lifecycle at rank zero. Service ranks interpret the registration payloads."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


def validate_prebuild(entries, roles):
    if not isinstance(entries, list):
        raise ValueError("prebuild must be an array")
    pairs = {}
    for item in entries:
        if (
            not isinstance(item, dict)
            or set(item) - {"prefill", "decode", "device_pairs"}
            or not all(
                isinstance(item.get(role), str) and roles.get(item[role]) == role for role in ("prefill", "decode")
            )
        ):
            raise ValueError("each prebuild entry requires a Prefill and a Decode ID")
        key = item["prefill"], item["decode"]
        devices = item.get("device_pairs", [])
        if (
            key in pairs
            or not isinstance(devices, list)
            or any(
                not isinstance(pair, list) or len(pair) != 2 or any(type(rank) is not int or rank < 0 for rank in pair)
                for pair in devices
            )
        ):
            raise ValueError("duplicate prebuild pair or invalid device rank pairs")
        if len({tuple(pair) for pair in devices}) != len(devices):
            raise ValueError("duplicate device rank pairs")
        pairs[key] = devices
    return pairs


@dataclass
class _Pair:
    state: str = "retired"
    id: str | None = None
    phase: str | None = None
    deadline: float | None = None
    receiver_ready: bool = False
    domain_bound: bool = False
    device_connected: bool | None = None
    error: str | None = None


class PairDirectory:
    def __init__(self, identities, prebuild, timeout, send):
        self.identities = identities
        self.ranks = {value["id"]: rank for rank, value in identities.items()}
        self.prebuild = validate_prebuild(prebuild, {s["id"]: s["role"] for s in identities.values()})
        self.registrations = {}
        self.pairs = {
            (p["id"], d["id"]): _Pair()
            for p in identities.values()
            if p["role"] == "prefill"
            for d in identities.values()
            if d["role"] == "decode"
        }
        self.timeout, self.send = timeout, send
        self.draining = False

    def snapshot(self):
        instances = []
        for rank, identity in self.identities.items():
            registered = self.registrations.get(identity["id"])
            if registered is not None:
                node = registered["descriptor"]
                instances.append({**identity, "rank": rank, "boot": node["boot"], "registry_digest": node["registry"]})
        return {
            "type": "snapshot",
            "draining": self.draining,
            "instances": instances,
            "pairs": [
                dict(
                    prefill=p,
                    decode=d,
                    state=pair.state,
                    domain_bound=pair.domain_bound,
                    receiver_ready=pair.receiver_ready,
                    device_connected=pair.device_connected,
                    error=pair.error,
                )
                for (p, d), pair in self.pairs.items()
            ],
        }

    @staticmethod
    def _registration(identity, payload):
        descriptor, registry = payload.get("descriptor"), payload.get("registry")
        if (
            not isinstance(descriptor, dict)
            or not isinstance(registry, str)
            or not registry
            or descriptor.get("node_id") != identity["id"]
            or descriptor.get("role") != identity["role"]
            or descriptor.get("healthy") is not True
            or not isinstance(descriptor.get("contract"), dict)
            or not all(isinstance(payload.get(k), str) and payload[k] for k in ("boot", "registry_digest"))
            or (descriptor.get("boot"), descriptor.get("registry")) != (payload["boot"], payload["registry_digest"])
        ):
            raise ValueError("registration differs from its rank identity or generation")
        return {"descriptor": descriptor, "registry": registry}

    def status(self, rank, state, payload):
        name = self.identities[rank]["id"]
        if self.draining:
            if state != "ready":
                self.registrations.pop(name, None)
            return
        old = self.registrations.pop(name, None)
        if state == "ready":
            try:
                self.registrations[name] = self._registration(self.identities[rank], payload)
            except ValueError as exc:
                print(f"MPI registration rejected for {name}: {exc}", flush=True)
        if old != self.registrations.get(name):
            for key in self.pairs:
                if name in key:
                    self._refresh(key)

    def _refresh(self, key):
        pair = self.pairs[key] = _Pair()
        if not all(name in self.registrations for name in key):
            return
        p, d = (self.registrations[name]["descriptor"] for name in key)
        if any(p.get(field) != d.get(field) for field in ("contract", "provider", "version")):
            pair.state, pair.error = "failed", "incompatible service descriptors"
        elif key not in self.prebuild:
            pair.state = "on_demand"
        else:
            pair.state = "preparing"
            self._command(key, pair, "decode")

    def _command(self, key, pair, role):
        pair.id, pair.phase = secrets.token_hex(16), role
        pair.deadline = time.monotonic() + self.timeout
        p, d = key
        cmd = {
            "type": "prepare_pair",
            "id": pair.id,
            "prefill": self.registrations[p],
            "decode": self.registrations[d],
            "device_pairs": self.prebuild[key],
            "receiver_ready": pair.receiver_ready,
            "timeout_s": self.timeout,
        }
        self.send(self.ranks[p if role == "prefill" else d], "prepare_pair", pair.id, cmd)

    def result(self, rank, result):
        for key, pair in self.pairs.items():
            if pair.id is None or pair.id != result.get("id"):
                continue
            expected = self.ranks[key[0 if pair.phase == "prefill" else 1]]
            if rank != expected:
                return
            if pair.deadline is not None and pair.deadline <= time.monotonic():
                self._fail(pair, "prebuild execution timed out")
                return
            node = self.registrations[self.identities[rank]["id"]]["descriptor"]
            if (result.get("boot"), result.get("registry_digest")) != (node["boot"], node["registry"]):
                return
            requested = bool(self.prebuild[key])
            if pair.phase == "decode":
                pair.receiver_ready = result.get("receiver_ready") is True
                if result.get("ok") is True and (not requested or pair.receiver_ready):
                    self._command(key, pair, "prefill")
                    return
            else:
                pair.domain_bound = result.get("domain_bound") is True
                pair.device_connected = result.get("device_connected") is True if requested else None
                if result.get("ok") is True and pair.domain_bound and (not requested or pair.device_connected):
                    pair.state = "prebuilt"
                    pair.id = pair.deadline = None
                    return
            self._fail(pair, str(result.get("error") or "incomplete prebuild result")[:1024])
            return

    @staticmethod
    def _fail(pair, error):
        pair.state, pair.error = "failed", error
        pair.id = pair.deadline = None

    def next_deadline(self, default):
        deadlines = [pair.deadline for pair in self.pairs.values() if pair.deadline is not None]
        if default is not None:
            deadlines.append(default)
        return min(deadlines, default=None)

    def expire(self):
        changed = False
        for pair in self.pairs.values():
            if pair.deadline is not None and pair.deadline <= time.monotonic():
                self._fail(pair, "prebuild execution timed out")
                changed = True
        return changed

    def drain(self):
        self.draining = True
        for pair in self.pairs.values():
            pair.state = "retired"
            pair.id = pair.deadline = None
