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
            "__thread int value = 7;",
            "static __thread int value = 7;",
            "_Thread_local int value = 7;",
            "static _Thread_local int value = 7;",
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

    def test_rejects_the_former_recorder_allowlist(self):
        path = "src/common/host_build_graph/host/graph_submit.cpp"
        declaration = "thread_local GraphRecording *g_active_graph_recording = nullptr;"
        result = self.run_checker(declaration, path)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("new thread_local in a reloadable SO", result.stdout)

    def test_rejects_unreviewed_key_creation(self):
        result = self.run_checker("void f() { pthread_key_create(&key, nullptr); }\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("src/probe.cpp: pthread key creation sites: 1; reviewed: 0", result.stdout)

    def test_accepts_the_reviewed_key_count(self):
        path = "src/common/log/host_log.cpp"
        result = self.run_checker("void f() { pthread_key_create(&key, std::free); }\n", path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_an_extra_key_in_a_reviewed_file(self):
        path = "src/common/log/host_log.cpp"
        source = "void f() { pthread_key_create(&a, nullptr); pthread_key_create(&b, nullptr); }\n"
        result = self.run_checker(source, path)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(
            "src/common/log/host_log.cpp: pthread key creation sites: 2; reviewed: 1",
            result.stdout,
        )

    def test_ignores_key_calls_in_comments_and_literals(self):
        source = (
            "// pthread_key_create(&key, nullptr);\n"
            'const char *text = "pthread_key_create";\n'
            "/* pthread_key_create(&key, nullptr); */\n"
        )
        result = self.run_checker(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
