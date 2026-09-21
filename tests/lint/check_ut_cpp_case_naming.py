# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Keep `test_*.cpp` under tests/ut/cpp/ meaning "a test case", and nothing else.

The C++ unit tests are discovered by a recursive glob for `test_*.cpp`, so the name
is what decides whether a file becomes a ctest target. A support file that happens to
be named `test_*` is therefore built as a case of its own — it links without a
`main`, or it registers no test and passes vacuously, and either way the directory
listing stops telling you what the suite covers.

Two kinds of file live here and only one of them is a case:

- **Cases** carry `TEST(...)` or `TEST_F(...)`. They are named `test_*.cpp` and the
  glob builds one target each.
- **Support files** are linked into someone else's target: link-time stubs, the
  shared objects a dlopen test loads, arch-specific stub patches, fixtures. They go
  under a `support/` directory and are not named `test_*`. Which targets may reach
  one is decided by where that directory sits: `tests/ut/cpp/support/` is the whole
  tree's, `tests/ut/cpp/common/support/` is that subtree's, and a `support/` beside
  a directory's cases belongs to those cases.

This hook keeps the glob's one input honest, so the convention holds without each
contributor remembering it.
"""

import argparse
import re
import sys
from pathlib import Path

UT_CPP_ROOT = "tests/ut/cpp/"

# A case registers at least one gtest. Matched at the start of a line so a mention in
# a comment or a string does not count.
GTEST_REGISTRATION = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(", re.MULTILINE)

# Directories that hold support files rather than cases. One name, at whatever
# depth its files are shared from.
SUPPORT_DIRS = ("support",)


def is_support_path(relative: Path) -> bool:
    return any(part in SUPPORT_DIRS for part in relative.parts)


def registers_a_test(path: Path) -> bool:
    try:
        return GTEST_REGISTRATION.search(path.read_text(encoding="utf-8")) is not None
    except (OSError, UnicodeDecodeError):
        # Unreadable files are not this hook's business; other hooks will complain.
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to check (pre-commit passes the staged set)")
    args = parser.parse_args()

    named_but_not_a_case = []
    case_in_support_dir = []

    for name in args.files:
        if not name.startswith(UT_CPP_ROOT) or not name.endswith(".cpp"):
            continue
        path = Path(name)
        relative = Path(name[len(UT_CPP_ROOT) :])
        named_as_case = path.name.startswith("test_")

        if named_as_case and is_support_path(relative):
            case_in_support_dir.append(name)
        elif named_as_case and not registers_a_test(path):
            named_but_not_a_case.append(name)

    if not named_but_not_a_case and not case_in_support_dir:
        return 0

    if named_but_not_a_case:
        print("These files are named test_*.cpp but register no gtest, so the recursive")
        print("glob would build each one as a test target of its own:")
        print()
        for name in named_but_not_a_case:
            print(f"  {name}")
        print()
        print("If the file is linked into another target — a stub, a dlopen'd shared")
        print("object, a fixture — move it under a support/ directory and drop the")
        print("test_ prefix. If it is a case, give it a TEST or TEST_F.")

    if case_in_support_dir:
        if named_but_not_a_case:
            print()
        print("These files are named test_*.cpp inside a support directory, which the")
        print("glob reads as a case and the directory says is not one:")
        print()
        for name in case_in_support_dir:
            print(f"  {name}")
        print()
        print("Drop the test_ prefix, or move the file out of support/.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
