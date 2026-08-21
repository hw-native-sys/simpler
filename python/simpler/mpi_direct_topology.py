# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validated static topology for the direct-MPI L4 control plane."""

from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .remote_l3_limits import MAX_FRAME_BYTES


def _positive_finite(value: Any, field: str) -> float:
    result = float(value)
    if not (result > 0.0 and math.isfinite(result)):
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class MpiDirectExecutorSpec:
    rank: int
    worker_id: int
    host: str
    platform: str
    runtime: str
    device_ids: tuple[int, ...]
    num_sub_workers: int
    comm_profile: str
    global_device_ranks: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MpiDirectExecutorSpec:
        spec = cls(
            rank=int(data["rank"]),
            worker_id=int(data["worker_id"]),
            host=str(data["host"]),
            platform=str(data["platform"]),
            runtime=str(data.get("runtime", "tensormap_and_ringbuffer")),
            device_ids=tuple(int(item) for item in data.get("device_ids", ())),
            num_sub_workers=int(data.get("num_sub_workers", 0)),
            comm_profile=str(data.get("comm_profile", "sim")),
            global_device_ranks=tuple(int(item) for item in data.get("global_device_ranks", ())),
        )
        if spec.rank <= 0:
            raise ValueError("executor rank must be positive; rank 0 is the controller")
        if spec.worker_id < 0:
            raise ValueError("executor worker_id must be non-negative")
        if not spec.host or not spec.platform or not spec.runtime or not spec.comm_profile:
            raise ValueError("executor host, platform, runtime, and comm_profile must be non-empty")
        if spec.num_sub_workers < 0:
            raise ValueError("executor num_sub_workers must be non-negative")
        if len(set(spec.device_ids)) != len(spec.device_ids) or any(device < 0 for device in spec.device_ids):
            raise ValueError("executor device_ids must be unique and non-negative")
        if spec.global_device_ranks and len(spec.global_device_ranks) != len(spec.device_ids):
            raise ValueError("executor global_device_ranks must match device_ids length")
        if len(set(spec.global_device_ranks)) != len(spec.global_device_ranks) or any(
            rank < 0 for rank in spec.global_device_ranks
        ):
            raise ValueError("executor global_device_ranks must be unique and non-negative")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "worker_id": self.worker_id,
            "host": self.host,
            "platform": self.platform,
            "runtime": self.runtime,
            "device_ids": list(self.device_ids),
            "num_sub_workers": self.num_sub_workers,
            "comm_profile": self.comm_profile,
            "global_device_ranks": list(self.global_device_ranks),
        }


@dataclass(frozen=True)
class MpiDirectTopology:
    controller_rank: int
    controller_host: str
    executors: tuple[MpiDirectExecutorSpec, ...]
    startup_timeout_s: float
    session_timeout_s: float
    heartbeat_interval_s: float
    max_pending_frame_bytes: int
    launcher_args: tuple[str, ...]

    @property
    def world_size(self) -> int:
        return 1 + len(self.executors)

    @property
    def hosts(self) -> tuple[str, ...]:
        return (self.controller_host,) + tuple(spec.host for spec in self.executors)

    def executor_for_rank(self, rank: int) -> MpiDirectExecutorSpec:
        if rank <= 0 or rank >= self.world_size:
            raise ValueError(f"MPI rank {rank} is outside executor range")
        spec = self.executors[rank - 1]
        if spec.rank != rank:
            raise RuntimeError("executor table is not rank ordered")
        return spec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MpiDirectTopology:
        parsed = tuple(MpiDirectExecutorSpec.from_dict(item) for item in data.get("executor_ranks", ()))
        explicit = {rank for spec in parsed for rank in spec.global_device_ranks}
        next_rank = 0
        executors_list = []
        for spec in parsed:
            if spec.global_device_ranks:
                executors_list.append(spec)
                continue
            assigned = []
            for _device_id in spec.device_ids:
                while next_rank in explicit:
                    next_rank += 1
                assigned.append(next_rank)
                explicit.add(next_rank)
                next_rank += 1
            executors_list.append(replace(spec, global_device_ranks=tuple(assigned)))
        executors = tuple(executors_list)
        topology = cls(
            controller_rank=int(data.get("controller_rank", 0)),
            controller_host=str(data.get("controller_host", "localhost")),
            executors=executors,
            startup_timeout_s=_positive_finite(data.get("startup_timeout_s", 60.0), "startup_timeout_s"),
            session_timeout_s=_positive_finite(data.get("session_timeout_s", 30.0), "session_timeout_s"),
            heartbeat_interval_s=_positive_finite(data.get("heartbeat_interval_s", 1.0), "heartbeat_interval_s"),
            max_pending_frame_bytes=int(data.get("max_pending_frame_bytes", 64 * 1024 * 1024)),
            launcher_args=tuple(str(arg) for arg in data.get("launcher_args", ())),
        )
        topology.validate()
        return topology

    @classmethod
    def load(cls, path: str) -> MpiDirectTopology:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("MPI direct topology root must be an object")
        return cls.from_dict(data)

    def validate(self) -> None:
        if self.controller_rank != 0:
            raise ValueError("PR3 requires controller_rank=0")
        if not self.controller_host:
            raise ValueError("controller_host must be non-empty")
        if not self.executors:
            raise ValueError("executor_ranks must contain at least one executor")
        if len(set(self.hosts)) > 1 and _is_loopback_host(self.controller_host):
            raise ValueError(
                f"controller_host={self.controller_host!r} is a loopback address, but the topology spans multiple "
                "hosts; set controller_host to an address reachable from all MPI hosts"
            )
        closed_hosts: set[str] = set()
        previous_host: str | None = None
        for host in self.hosts:
            if host != previous_host:
                if host in closed_hosts:
                    raise ValueError("topology hosts must be contiguous in rank order")
                if previous_host is not None:
                    closed_hosts.add(previous_host)
                previous_host = host
        if [spec.rank for spec in self.executors] != list(range(1, self.world_size)):
            raise ValueError("executor ranks must be dense and ordered from 1 to world_size-1")
        if [spec.worker_id for spec in self.executors] != list(range(len(self.executors))):
            raise ValueError("executor worker_ids must be dense and ordered from 0")
        all_global_ranks = [rank for spec in self.executors for rank in spec.global_device_ranks]
        if len(set(all_global_ranks)) != len(all_global_ranks):
            raise ValueError("global_device_ranks must be unique across all executors")
        if self.max_pending_frame_bytes < MAX_FRAME_BYTES:
            raise ValueError("max_pending_frame_bytes must fit one maximum SLR3 frame")

    def runtime_manifest(self, session_id: int) -> dict[str, Any]:
        if int(session_id) == 0:
            raise ValueError("session_id must be non-zero")
        return {
            "version": 1,
            "controller_rank": self.controller_rank,
            "controller_host": self.controller_host,
            "world_size": self.world_size,
            "session_id": int(session_id),
            "startup_timeout_s": self.startup_timeout_s,
            "session_timeout_s": self.session_timeout_s,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "max_pending_frame_bytes": self.max_pending_frame_bytes,
            "executor_ranks": [spec.to_dict() for spec in self.executors],
        }


def load_runtime_manifest(path: str) -> tuple[MpiDirectTopology, int]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or int(data.get("version", 0)) != 1:
        raise ValueError("unsupported MPI direct runtime manifest")
    return load_runtime_manifest_data(data)


def load_runtime_manifest_data(data: dict[str, Any]) -> tuple[MpiDirectTopology, int]:
    if not isinstance(data, dict) or int(data.get("version", 0)) != 1:
        raise ValueError("unsupported MPI direct runtime manifest")
    topology = MpiDirectTopology.from_dict(data)
    declared_world_size = int(data.get("world_size", 0))
    if declared_world_size != topology.world_size:
        raise ValueError("runtime manifest world_size does not match executor table")
    session_id = int(data.get("session_id", 0))
    if session_id == 0:
        raise ValueError("runtime manifest session_id must be non-zero")
    return topology, session_id
