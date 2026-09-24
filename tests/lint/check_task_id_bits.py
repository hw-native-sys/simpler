# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep task_id bit-field knowledge out of the runtime-agnostic DFX tools.

A task id carries whichever TaskId layout its runtime uses, and the two layouts
disagree about what the high bits mean: host_build_graph reads bits 63:62 as an
id space and 51:32 as a parent id, tensormap_and_ringbuffer reads 63:32 as a
ring index. Nothing in the value says which. So a shift or mask written in
common code is a silent bet on one runtime, and when the bet is wrong the
result still looks like an id -- an hbg sub-task decoded as tmr becomes a
plausible ``r3t5`` with a billion-scale ring.

The Python mirrors of those layouts live in ``tools/hbg/task_id.py`` and
``tools/tmr/task_id.py``; every other tool asks one of them. This hook holds
that line, because the symptom of losing it is not a crash but a label: the
regression that prompted the split rendered ids as ``T1073741827_0``.

Why the patterns are narrow rather than "any bit twiddling":

  Common DFX code legitimately shifts and masks for things that are not task
  ids -- bf16-to-float conversion (``raw << 16``), a tensor id's short form
  (``tid & 0xFFFF``), unsigned 64-bit coercion (``1 << 64``), and a CCE kernel
  launch (``<<<1>>>``) in generated source. A hook that flagged those would be
  turned off, so only the constants the two layouts actually use are listed.
  A new layout constant in either ``task_id.h`` belongs here too.

Why the exemption names a line rather than counting them:

  ``_runtime_dispatch.raw_halves()`` splits a raw id at bit 32 to build a
  DOT-safe node name. That is not a decode -- it makes no claim about what
  either half means, which is what lets it stay correct for a corrupt record
  where ``local_id()`` would not -- so it is allowed. The allowance matches that
  line's text, so a real decode added anywhere in the file is still reported,
  and reported at its own line. A count would instead be spent by whichever
  match came first, blaming ``raw_halves`` for a decode written above it.

This file is outside the checked set, because the patterns have to be written
here to be enforced and quoted here to be explained -- the same reason
``check_retired_names.py`` exempts itself. It is not a DFX tool, so nothing in
it could decode a task id in the first place.
"""

import argparse
import re
import sys
from pathlib import Path

# The shifts and masks the two TaskId layouts are built from. Spacing is free
# because both `raw >> 32` and `raw>>32` are the same bet; the hex masks are
# length-anchored so 0xFFFFF (a 20-bit parent) does not also match 0xFFFF (a
# tensor id's short form) or 0xFFFFFF.
LAYOUT_BITS = re.compile(
    r"""
    >>\s*(?:62|32)\b      # space / parent / ring extraction
  | <<\s*(?:62|32)\b      # the same fields, being composed
  | 0x[fF]{5}\b           # PARENT_BITS = 20
  | 0x[fF]{8}\b           # the low word
    """,
    re.VERBOSE,
)

# Lines allowed to match, per file, and why. Keyed by the line's exact stripped
# text so the allowance cannot drift onto a different line. See the module
# docstring: this is an allowance for one known non-decode, not a licence.
ALLOWED_LINES = {
    "simpler_setup/tools/_runtime_dispatch.py": frozenset(
        {
            "return raw >> 32, raw & 0xFFFFFFFF",  # raw_halves()
        }
    ),
}

_NOTHING_ALLOWED: "frozenset[str]" = frozenset()


def matching_lines(path: Path, allowed: "frozenset[str]") -> "list[tuple[int, str]]":
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [
        (lineno, stripped)
        for lineno, line in enumerate(text.splitlines(), 1)
        if LAYOUT_BITS.search(line) and (stripped := line.strip()) not in allowed
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to check (pre-commit passes the staged set)")
    args = parser.parse_args()

    failures = []
    for name in args.files:
        for lineno, line in matching_lines(Path(name), ALLOWED_LINES.get(name, _NOTHING_ALLOWED)):
            failures.append(f"{name}:{lineno}: {line}")

    if not failures:
        return 0

    print("A task_id layout is being decoded outside the runtime that owns it.")
    print()
    print("The two layouts disagree about the high bits, and the value does not say")
    print("which it follows, so a shift or mask here silently assumes one runtime.")
    print("Resolve the capture's runtime and ask it instead:")
    print()
    print("    from simpler_setup.tools._runtime_dispatch import get")
    print("    fields = get(runtime_name).task_id_fields(task_id)")
    print("    label = get(runtime_name).display(task_id)")
    print()
    print("For an id you only need to normalize, not decode, use")
    print("`_runtime_dispatch.normalize_task_id_int`, which is runtime-agnostic.")
    print()
    for failure in failures:
        print(f"  {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
