# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject compiler TLS in reloadable framework source; pthread ownership is explicit."""

import argparse
import re
from pathlib import Path

# Preserve newlines for diagnostics while ignoring comments and quoted literals.
NON_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
# Reviewed creation sites, counted after comments/literals are removed. New sites
# (including additional calls in an approved file) require an ownership review.
KEY_SITES = {
    "common/log/host_log.cpp": 1,  # Process logger; libc free callback.
    "common/platform/include/common/module_thread_value.h": 1,  # Module-owned scalar keys.
    "common/platform/sim/sim_context/cpu_sim_context.cpp": 3,  # Process-owned sim selectors.
    "common/platform/onboard/host/c_api_shared.cpp": 1,  # Module-owned prepare admission.
    "common/host_build_graph/host/dep_gen_host_graph.cpp": 2,  # Module-owned capture selectors.
    "a2a3/platform/sim/aicore/kernel.cpp": 6,  # Module-owned keys, stack-owned values.
    "a5/platform/sim/aicore/kernel.cpp": 1,
}
KEY_CREATE = re.compile(r"\bpthread_key_create\s*\(")
# Temporary exact declarations pending PR #2859; no other compiler TLS is allowed.
LEGACY_TLS = {
    "common/platform/onboard/aicpu/platform_aicpu_affinity.cpp": "static thread_local int32_t tl_exec_idx = -1;",
    "common/platform/sim/aicpu/platform_aicpu_affinity.cpp": "static thread_local int32_t tl_filter_exec_idx = -1;",
}
TLS = re.compile(r"\b(?:thread_local|__thread|_Thread_local)\b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "src")
    args = parser.parse_args()
    failures = []
    observed = {}
    for path in sorted(args.root.rglob("*")):
        if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cc"}:
            continue
        code = NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match[0]), path.read_text())
        count = len(KEY_CREATE.findall(code))
        if count:
            observed[path.relative_to(args.root).as_posix()] = count
        legacy = LEGACY_TLS.get(path.relative_to(args.root).as_posix())
        if legacy:
            code = code.replace(legacy, " " * len(legacy), 1)
        for match in TLS.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            failures.append(f"{path}:{line}: compiler TLS in reloadable framework code: {match[0]}")
    for name in sorted(observed.keys() | KEY_SITES.keys()):
        expected = KEY_SITES.get(name, 0)
        actual = observed.get(name, 0)
        if actual != expected:
            failures.append(f"{name}: pthread key creation sites: {actual}; reviewed: {expected}")
    for failure in failures:
        print(failure)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
