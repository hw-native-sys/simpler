# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host discovery of communication contexts passed to a CCU callable."""

from __future__ import annotations


def discover_ccu_domains(args, worker_id: int, domains):
    """Resolve explicit context scalars against live, owned domain generations.

    Tensor provenance is checked by ordinary submission; per-call Tensor ranges
    are checked by the CCU device interface. An L2 task may also carry tensors
    for ordinary computation, so its entire Tensor list is not a CCU signature.
    """
    scalars = {int(args.scalar(i)) for i in range(args.scalar_count())}
    selected = []
    addresses = set()
    for domain in domains:
        context = domain.contexts.get(worker_id)
        if context is None or not context.device_ctx or context.device_ctx not in scalars:
            continue
        if domain.released or domain.freed:
            raise ValueError("CCU submission names a released communication domain")
        config = domain._config
        if config is None or "ccu" not in config.workspace_engines:
            raise ValueError("CCU submission requires a domain allocated with workspace_engines=('ccu',)")
        if domain.generation <= 0 or not 0 <= domain.domain_id < 1 << 64:
            raise ValueError("CCU submission has an invalid domain identity")
        if context.device_ctx in addresses:
            raise ValueError("CCU device context ambiguously names multiple domains")
        addresses.add(context.device_ctx)
        selected.append(domain)
    if not selected:
        raise ValueError("CCU submission requires a live domain device_context scalar for the selected worker")
    return tuple(selected)


def require_ccu_execution_backend(target, *, native_provider: bool = False) -> None:
    """Prevent embedded CCU definitions from being silently ignored at launch."""
    if getattr(target, "ccu_function_count", 0) and not native_provider:
        raise RuntimeError(
            "CCU execution requires the callable/domain backend and hidden device context integration; "
            "this runtime supports embedded CCU definitions but cannot execute them"
        )
