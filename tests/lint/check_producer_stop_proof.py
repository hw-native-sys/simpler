#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the producer-stop proof stated where a reader can find it.

A failed diagnostic run's terminal bank goes back to its pool when, and only
when, nothing on the device can still write it. The two platforms answer that
differently, and this checks that each keeps saying what it can actually mean:

  * a **simulated** device's producers are this process's own threads, and the
    runner joins them before settling the run. That settlement goes through
    `settle_run_after_producer_stop`, which reads the proof that call
    produced and is covered by
    `tests/ut/cpp/common/platform/test_run_terminal_stop_settlement.cpp`;
  * a **real** device's producers do not become observable that way. The
    onboard runner base does not take the argument at all, so those arch
    runners cannot claim a stop even by mistake, and the base passes `false` to
    every close wrapper itself.

**What this is not.** It is a text check and it cannot prove control flow: it
cannot tell a join that runs from one in a branch that does not, and it is not
a substitute for executing the failure path. The executed coverage of the
decision is the unit suite named above, which calls the same production entry;
what remains unexecuted anywhere is the join itself, because reaching it needs
a simulated device — that translation unit dlopens the AICPU object and wants a
prepared execution to fail. This file's single job is narrower: that no arch
runner quietly starts claiming a proof its platform cannot make, and that a sim
settlement keeps going through the entry which carries one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The calls that carry the proof to a run boundary.
SETTLING_CALLS = (
    "teardown_shared_collectors_after_run",
    "emit_device_dep_gen_graph",
    "settle_run_after_producer_stop",
    # The per-owner boundaries the onboard base forwards to itself. They are
    # where that platform's answer is actually written, so they are read on the
    # same terms as the runners that cannot state one.
    "close_pmu_run_boundary",
    "close_args_dump_run_boundary",
    "close_scope_stats_run_boundary",
    "close_chip_swimlane_run_boundary",
)

# The entry that states the simulated platform's proof, and the only settling
# call that may do so without naming it at the call site.
JOINED_SETTLEMENT = "settle_run_after_producer_stop"


def settling_sites() -> list[tuple[Path, bool]]:
    """Every file that states the proof, paired with whether it may claim one."""
    found: list[tuple[Path, bool]] = []
    for arch in ("a2a3", "a5"):
        for backend, simulated in (("sim", True), ("onboard", False)):
            path = REPO_ROOT / "src" / arch / "platform" / backend / "host" / "device_runner.cpp"
            if path.exists():
                found.append((path, simulated))
    # The onboard base is where that platform's answer is actually written, so
    # it is held to the same rule as the runners that cannot state one.
    base = REPO_ROOT / "src" / "common" / "platform" / "onboard" / "host" / "device_runner_base.cpp"
    if base.exists():
        found.append((base, False))
    return found


def strip_comments(text: str) -> str:
    """Blank out comments, keeping offsets so reported lines stay true.

    Argument comments (`/*device_producers_stopped=*/true`) are the one kind
    this keeps: they are how a literal is named at a call site, and dropping
    them would blind the check to the thing it reads.
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            if "device_producers_stopped=" not in text[i:end]:
                for j in range(i, end):
                    if out[j] != "\n":
                        out[j] = " "
            i = end
        elif text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            for j in range(i, end):
                out[j] = " "
            i = end
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
        else:
            i += 1
    return "".join(out)


def call_arguments(text: str, open_paren: int) -> str:
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    return ""


def is_declaration(args: str) -> bool:
    """A declaration or definition names parameter types; a call does not."""
    return "bool device_producers_stopped" in args


def check_file(path: Path, simulated: bool) -> list[str]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    where = path.relative_to(REPO_ROOT)
    problems: list[str] = []
    for call in SETTLING_CALLS:
        for match in re.finditer(r"(?<![\w:>.])" + re.escape(call) + r"\s*\(", text):
            args = call_arguments(text, match.end() - 1)
            if is_declaration(args):
                continue
            line = text.count("\n", 0, match.start()) + 1
            stated = re.search(r"/\*device_producers_stopped=\*/\s*(true|false)", args)
            claims_stopped = stated is not None and stated.group(1) == "true"

            if call == JOINED_SETTLEMENT:
                # This entry carries the proof itself, so a call site states
                # nothing — but only a platform that can make the proof may
                # reach it at all.
                if not simulated:
                    problems.append(
                        f"{where}:{line}: an onboard path settles through {JOINED_SETTLEMENT}; the stop it reads is "
                        "a join this platform's producers never have"
                    )
                continue

            if not simulated:
                if claims_stopped:
                    problems.append(
                        f"{where}:{line}: an onboard path claims its device producers stopped; nothing there "
                        "observes them, and the bank must be withdrawn instead"
                    )
                continue

            if call.startswith("close_") and simulated:
                # The sim base forwards the caller's own value here, which is a
                # computed proof and a correct one; nothing to judge.
                continue

            if call == "teardown_shared_collectors_after_run" or call.startswith("close_"):
                # A simulated settlement has a named entry that carries the
                # proof and a case that calls it; stating `true` here instead
                # puts the decision back where nothing executes it.
                if claims_stopped:
                    problems.append(
                        f"{where}:{line}: a simulated runner states the stop proof at a call site; it belongs "
                        f"in {JOINED_SETTLEMENT}, which reads the stop that was actually performed"
                    )
                continue

            # The dep_gen emit is the arch runner's own and takes the proof as a
            # parameter on both backends, so the call site is the only place a
            # simulated runner can state it — and it must, because leaving it to
            # a default hides which platform's answer this is.
            if stated is None:
                problems.append(
                    f"{where}:{line}: {call} states no device_producers_stopped; this platform has a proof to "
                    "give and leaving it to a default hides which one"
                )
            elif not claims_stopped:
                problems.append(
                    f"{where}:{line}: a simulated runner withdraws this run's bank; its producers are this "
                    "process's threads and the settlement beside this one proves they stopped"
                )
    return problems


def main() -> int:
    sites = settling_sites()
    if not sites:
        print("✗ no device runner was found; this check cannot have passed")
        return 1

    problems: list[str] = []
    for path, simulated in sites:
        problems.extend(check_file(path, simulated))

    if problems:
        print("✗ the producer-stop proof is not stated where it can be relied on:\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"✓ every settling call in {len(sites)} files states a proof its platform can make")
    return 0


if __name__ == "__main__":
    sys.exit(main())
