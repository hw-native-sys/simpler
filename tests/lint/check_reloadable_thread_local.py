#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep new C++ thread_local declarations out of the reloadable SO source tree.

docs/dynamic-linking.md forbids C++ thread_local in any SO that is dlclosed and
re-dlopen'd: ELF TLSDESC state can outlive the image, and a thread that survives
the unload keeps the stale slot. The device-side AICore/AICPU code now carries
its per-thread state through module-owned pthread keys, and the remaining
host-side declarations are the documented exceptions listed below. Recorder and
capture TLS ownership is tracked separately in #2915.

The allowlist is the inventory: a new declaration anywhere under `src/` fails
unless its exact line is added here and reviewed. Changing one of the listed
lines fails too, so the inventory cannot drift silently. The check recognizes
the keyword outside comments and literals, regardless of declaration order or
attributes. It does not expand macros or evaluate preprocessor conditions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Comments and literals consume their contents before a keyword can match.
# Numeric tokens consume digit separators, which are not character quotes.
TOKEN = re.compile(
    r"//(?:\\\r?\n|[^\n])*"
    r"|/\*.*?\*/"
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"'
    r"|\b[0-9][\w.']*"
    r"|'(?:\\.|[^'\\])*'"
    r"|(?P<keyword>\bthread_local\b)",
    re.DOTALL,
)

# Documented exceptions. The key is the repo-relative path; the value is the
# exact stripped declaration line. Recorder/capture ownership is #2915; the
# workspace-plan and admission-claim state is host-only.
ALLOWED = {
    "src/common/platform/onboard/host/device_runner_base.cpp": {
        "thread_local WorkspacePlanIdentity g_workspace_plan;",
    },
    "src/common/platform/include/host/auto_admission.h": {
        "static thread_local std::vector<Claim> live;",
    },
    "src/common/host_build_graph/host/graph_submit.cpp": {
        "thread_local GraphInflightRecording *g_active_graph_entry = nullptr;",
        "thread_local GraphRecording *g_active_graph_recording = nullptr;",
        "thread_local GraphHostState *g_active_graph_owner = nullptr;",
    },
    "src/common/host_build_graph/host/dep_gen_host_graph.cpp": {
        "static thread_local HostGraphState s;",
    },
    "src/common/host_build_graph/host/graph_recording.cpp": {
        "static thread_local GraphRecording storage;",
    },
}


def main(argv: list[str]) -> int:
    failed = False
    for raw in argv[1:]:
        path = Path(raw)
        key = path.as_posix()
        allowed = ALLOWED.get(key, set())
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            print(f"{path}: cannot read: {error}")
            failed = True
            continue
        lines = text.splitlines()
        for token in TOKEN.finditer(text):
            if token.group("keyword") is None:
                continue
            number = text.count("\n", 0, token.start()) + 1
            if lines[number - 1].strip() in allowed:
                continue
            print(
                f"{path}:{number}: new thread_local in a reloadable SO; "
                f"use pthread TLS or add the exact declaration to the documented "
                f"inventory (see docs/dynamic-linking.md)"
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
