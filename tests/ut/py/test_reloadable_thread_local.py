# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the reloadable TLS hook through its command-line interface."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[2] / "lint/check_reloadable_thread_local.py"


class ReloadableThreadLocalTest(unittest.TestCase):
    def run_checker(self, source: str, path: str = "src/probe.cpp") -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(CHECKER), path],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )

    def test_rejects_storage_specifier_positions_and_attributes(self):
        for source in (
            "thread_local int value = 7;",
            "const thread_local int value = 7;",
            "static const thread_local int value = 7;",
            "int thread_local value = 7;",
            "[[maybe_unused]] thread_local int value = 7;",
            "alignas(64) thread_local int value = 7;",
            "void f() { static thread_local int value = 7; }",
            "namespace detail { thread_local int value = 7; }",
            "static /* storage */ thread_local int value = 7;",
        ):
            with self.subTest(source=source):
                result = self.run_checker(source)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("src/probe.cpp:1:", result.stdout)

    def test_ignores_comments_literals_and_longer_identifiers(self):
        for source in (
            "// thread_local int example;\nint value;",
            "/*\nthread_local int example;\n*/\nint value;",
            '// "quote" /* thread_local int example;\nint value;',
            'const char *message = "thread_local";',
            r'const char *message = "escaped \" thread_local";',
            'const char *message = R"tag(\nthread_local int example;\n)tag";',
            'const char *message = u8R"tag(\nthread_local int example;\n)tag";',
            "char quote = '\\''; // thread_local int example;",
            "int thread_local_count = 0; int not_thread_local = 0;",
        ):
            with self.subTest(source=source):
                result = self.run_checker(source)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")

    def test_preserves_diagnostic_lines_after_multiline_literals(self):
        source = 'auto text = R"tag(\nthread_local is text\n)tag";\nconst thread_local int value = 7;\n'
        result = self.run_checker(source)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("src/probe.cpp:4:", result.stdout)
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_digit_separators_do_not_hide_declarations(self):
        result = self.run_checker("auto first = 1'000;\nconst thread_local int value = 7;\nauto second = 2'000;\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("src/probe.cpp:2:", result.stdout)

    def test_allowlist_requires_the_exact_path_and_line(self):
        path = "src/common/host_build_graph/host/graph_submit.cpp"
        allowed = "thread_local GraphRecording *g_active_graph_recording = nullptr;"
        result = self.run_checker(allowed, path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_checker(allowed)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        result = self.run_checker(allowed.replace("nullptr", "other_recording"), path)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
