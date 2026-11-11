#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep reloadable SO TLS ownership explicit and reviewed.

docs/dynamic-linking.md forbids C++ thread_local in any SO that is dlclosed and
re-dlopen'd: ELF TLSDESC state can outlive the image, and a thread that survives
the unload keeps the stale slot. Per-thread state now lives in module-owned
pthread keys instead. This check enforces both halves of that contract:

- no compiler TLS declaration anywhere under `src/` unless its exact line is in
  the documented inventory below (empty today; the inventory stays as the
  reviewable extension point);
- the number of `pthread_key_create` sites in each source file matches the
  reviewed inventory, so a new key owner, an additional call in an approved
  file, or a silent removal all require an ownership review.

The scan recognizes keywords outside comments and literals, regardless of
declaration order or attributes. It does not expand macros or evaluate
preprocessor conditions. When run with explicit paths (as pre-commit file lists
do) only those files are checked; with no arguments it scans `src/` and also
reports inventory entries whose key sites disappeared.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SOURCE_SUFFIXES = {".h", ".hpp", ".c", ".cpp", ".cc", ".cxx"}

# Comments and literals consume their contents before a keyword can match.
# Numeric tokens consume digit separators, which are not character quotes.
TOKEN = re.compile(
    r"//(?:\\\r?\n|[^\n])*"
    r"|/\*.*?\*/"
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"'
    r"|\b[0-9][\w.']*"
    r"|'(?:\\.|[^'\\])*'"
    r"|(?P<keyword>\b(?:thread_local|__thread|_Thread_local)\b)",
    re.DOTALL,
)

KEY_CREATE = re.compile(r"\bpthread_key_create\s*\(")

# Documented exceptions. The key is the repo-relative path; the value is the
# exact stripped declaration line. There are none today: #2915 moved recorder
# and prepare state out of compiler TLS. Add a line only for a reviewed,
# documented exception.
ALLOWED: dict[str, set[str]] = {}

# Reviewed pthread key creation sites. The key is the repo-relative path and
# the value is the expected number of `pthread_key_create` calls. Every entry
# below names its owner in the docs/dynamic-linking.md inventory.
KEY_SITES = {
    "src/common/log/host_log.cpp": 1,  # Process logger; libc free callback.
    "src/common/platform/include/common/module_thread_value.h": 1,  # Module-owned scalar keys.
    "src/common/platform/include/aicpu/affinity_thread_state.h": 1,  # Module-owned affinity records.
    "src/common/platform/sim/sim_context/cpu_sim_context.cpp": 3,  # Process-owned sim selectors.
    "src/common/platform/onboard/host/c_api_shared.cpp": 1,  # Module-owned prepare admission.
    "src/a2a3/platform/sim/aicore/kernel.cpp": 1,  # Module-owned keys, stack-owned values.
    "src/a5/platform/sim/aicore/kernel.cpp": 1,  # Module-owned keys, stack-owned values.
    "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_scalar_record.h": 1,  # Module-owned scalar binding.
}


def without_comments_and_literals(text: str) -> str:
    """Blank every non-keyword token so scans see code only, preserving lines."""
    parts: list[str] = []
    last = 0
    for match in TOKEN.finditer(text):
        if match.group("keyword") is not None:
            continue
        parts.append(text[last : match.start()])
        parts.append(re.sub(r"[^\n]", " ", match.group(0)))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def check_file(path: Path, key: str, observed: dict[str, int]) -> list[str]:
    failures: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return [f"{path}: cannot read: {error}"]
    allowed = ALLOWED.get(key, set())
    lines = text.splitlines()
    for token in TOKEN.finditer(text):
        if token.group("keyword") is None:
            continue
        number = text.count("\n", 0, token.start()) + 1
        if lines[number - 1].strip() in allowed:
            continue
        failures.append(
            f"{path}:{number}: new {token.group('keyword')} in a reloadable SO; "
            f"use pthread TLS or add the exact declaration to the documented "
            f"inventory (see docs/dynamic-linking.md)"
        )
    observed[key] = len(KEY_CREATE.findall(without_comments_and_literals(text)))
    return failures


def source_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix in SOURCE_SUFFIXES)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [(Path(raw), Path(raw).as_posix()) for raw in argv[1:]]
        scan_all = False
    else:
        repo_root = Path(__file__).resolve().parents[2]
        targets = [(path, path.relative_to(repo_root).as_posix()) for path in source_files(repo_root / "src")]
        scan_all = True

    failures: list[str] = []
    observed: dict[str, int] = {}
    for path, key in targets:
        failures.extend(check_file(path, key, observed))

    reviewed = set(observed) | (set(KEY_SITES) if scan_all else set())
    for name in sorted(reviewed):
        expected = KEY_SITES.get(name, 0)
        actual = observed.get(name, 0)
        if actual != expected:
            failures.append(f"{name}: pthread key creation sites: {actual}; reviewed: {expected}")

    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
