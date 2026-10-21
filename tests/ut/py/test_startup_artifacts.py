# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free behavior tests; load the helper without native Simpler imports."""

import dataclasses
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import py_compile
import sys
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[3] / "python/simpler/startup_artifacts.py"


def archive_bytes(entries=None):
    if entries is None:
        entries = [("entry.py", b"def answer():\n    return 42\n", tarfile.REGTYPE)]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            info.linkname = "entry.py" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
            archive.addfile(info, io.BytesIO(data) if info.size else None)
    return gzip.compress(stream.getvalue(), mtime=0)


class StartupArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE_PATH.is_file():
            spec = importlib.util.spec_from_file_location("_startup_artifacts_under_test", MODULE_PATH)
            cls.api = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.api
            spec.loader.exec_module(cls.api)

    @classmethod
    def tearDownClass(cls):
        api = getattr(cls, "api", None)
        if api is not None:
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not api._FINDER]
            if sys.modules.get(api.__name__) is api:
                del sys.modules[api.__name__]

    def setUp(self):
        self.assertTrue(MODULE_PATH.is_file(), "startup artifact helper is not implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = "startup_artifact_test_" + self.id().split(".")[-1]
        self.addCleanup(self.clear_modules)

    def clear_modules(self):
        for name in list(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + "."):
                del sys.modules[name]
        for prefix in list(self.api._FINDER.bindings):
            if prefix == self.prefix or prefix.startswith(self.prefix + "."):
                del self.api._FINDER.bindings[prefix]

    def description(self, payload=None, prefix=None):
        payload = archive_bytes() if payload is None else payload
        return {
            "protocol_version": 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "module_prefix": prefix or self.prefix,
            "payload": payload,
        }

    def install(self, payload=None, prefix=None):
        return self.api.install_startup_artifacts([self.description(payload, prefix)], self.root / "cache")[0]

    def test_spec_is_frozen_json_safe_and_validates_identity(self):
        digest = "a" * 64
        spec = self.api.StartupArtifactSpec(self.root / "a.tar.gz", digest, "bundle.generated")
        self.assertEqual(
            json.loads(json.dumps(spec.to_manifest())),
            {
                "protocol_version": 1,
                "archive_path": str(self.root / "a.tar.gz"),
                "sha256": digest,
                "module_prefix": "bundle.generated",
            },
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.sha256 = "b" * 64
        for invalid in ("A" * 64, "a" * 63, "g" * 64, "../x"):
            with self.subTest(digest=invalid), self.assertRaises(ValueError):
                self.api.StartupArtifactSpec("a.tar.gz", invalid, "bundle")
        for invalid in ("", "../bundle", "bundle..entry", "a/b", "a\\b", "1bad", "class"):
            with self.subTest(prefix=invalid), self.assertRaises(ValueError):
                self.api.StartupArtifactSpec("a.tar.gz", digest, invalid)

    def test_reader_transports_bytes_and_receiver_never_needs_source_path(self):
        payload = archive_bytes()
        source = self.root / "source.tar.gz"
        source.write_bytes(payload)
        spec = self.api.StartupArtifactSpec(source, hashlib.sha256(payload).hexdigest(), self.prefix)
        descriptions = self.api.read_startup_artifacts([spec.to_manifest()])
        self.assertEqual(descriptions, [self.description(payload)])
        source.unlink()
        installed = self.api.install_startup_artifacts(descriptions, self.root / "cache")[0]
        self.assertEqual((installed.root / "entry.py").read_bytes(), b"def answer():\n    return 42\n")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            installed.root = self.root

    def test_tampered_digest_rejected_by_reader_and_installer(self):
        payload = archive_bytes()
        source = self.root / "source.tar.gz"
        source.write_bytes(payload)
        spec = self.api.StartupArtifactSpec(source, "0" * 64, self.prefix)
        with self.assertRaisesRegex(ValueError, "digest|SHA"):
            self.api.read_startup_artifacts([spec.to_manifest()])
        description = dict(self.description(payload), sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "digest|SHA"):
            self.api.install_startup_artifacts([description], self.root / "cache")

    def test_cache_hit_preserves_files_and_prefix_is_not_cache_identity(self):
        first = self.install()
        modified = (first.root / "entry.py").stat().st_mtime_ns
        second = self.install(prefix=self.prefix + "_alias")
        self.assertEqual(first.root, second.root)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual((second.root / "entry.py").stat().st_mtime_ns, modified)
        self.assertEqual(second.module_prefix, self.prefix + "_alias")

    def test_corrupted_cache_content_or_identity_is_rejected(self):
        installed = self.install()
        (installed.root / "entry.py").write_text("broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache|corrupt"):
            self.install()

    def test_unexpected_cache_file_is_rejected(self):
        installed = self.install()
        (installed.root / "unexpected.py").write_text("pass", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache|corrupt"):
            self.install()

    def test_corrupted_cache_identity_is_rejected(self):
        installed = self.install()
        (installed.root.parent / "identity.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache|corrupt"):
            self.install()

    def test_malformed_archives_are_rejected(self):
        for payload in (b"not gzip", gzip.compress(b"not tar", mtime=0)):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.install(payload)

    def test_rejects_pax_extension_chain_before_recursive_tar_parsing(self):
        extension = tarfile.TarInfo("pax")
        extension.type = tarfile.XHDTYPE
        raw = extension.tobuf(format=tarfile.USTAR_FORMAT) * 600
        raw += gzip.decompress(archive_bytes())
        with self.assertRaisesRegex(ValueError, "extension|USTAR|unsupported"):
            self.install(gzip.compress(raw, mtime=0))

    def test_rejects_each_pax_and_gnu_extension_header(self):
        for kind in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            extension = tarfile.TarInfo("extension")
            extension.type = kind
            raw = extension.tobuf(format=tarfile.USTAR_FORMAT) + gzip.decompress(archive_bytes())
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "extension|USTAR|unsupported"):
                self.install(gzip.compress(raw, mtime=0))

    def test_protocol_is_exact_and_malformed_descriptions_fail(self):
        good = self.description()
        for changes in (
            {"protocol_version": 2},
            {"protocol_version": True},
            {"payload": "not bytes"},
            {"module_prefix": "../bad"},
            {"archive_path": "parent-only.tar.gz"},
        ):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                self.api.install_startup_artifacts([dict(good, **changes)], self.root / "cache")
        for missing in good:
            bad = dict(good)
            del bad[missing]
            with self.subTest(missing=missing), self.assertRaises((ValueError, TypeError)):
                self.api.install_startup_artifacts([bad], self.root / "cache")

    def test_rejects_unsafe_archive_members(self):
        for name in (
            "../escape",
            "/absolute",
            "nested/../../escape",
            "a\\b",
            "C:/drive",
            "a//b",
            "./entry.py",
            "a/../entry.py",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.install(archive_bytes([(name, b"x", tarfile.REGTYPE)]))
        self.assertFalse((self.root / "escape").exists())
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.install(archive_bytes([("unsafe", b"", kind)]))

    def test_rejects_duplicates_and_conflicting_directory_paths(self):
        for entries in (
            [("entry.py", b"a", tarfile.REGTYPE), ("entry.py", b"b", tarfile.REGTYPE)],
            [("parent", b"a", tarfile.REGTYPE), ("parent/child", b"b", tarfile.REGTYPE)],
            [("parent/child", b"b", tarfile.REGTYPE), ("parent", b"a", tarfile.REGTYPE)],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.install(archive_bytes(entries))

    def test_size_and_member_limits_are_enforced(self):
        payload = archive_bytes()
        source = self.root / "oversize.tar.gz"
        source.write_bytes(payload)
        spec = self.api.StartupArtifactSpec(source, hashlib.sha256(payload).hexdigest(), self.prefix)
        with patch.object(self.api, "MAX_PAYLOAD_BYTES", len(payload) - 1):
            with self.assertRaisesRegex(ValueError, "limit|large"):
                self.api.read_startup_artifacts([spec.to_manifest()])
            with self.assertRaisesRegex(ValueError, "limit|large"):
                self.install(payload)
        with patch.object(self.api, "MAX_UNPACKED_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "limit|large"):
                self.install(payload)
        entries = [(str(i), b"x", tarfile.REGTYPE) for i in range(3)]
        with patch.object(self.api, "MAX_MEMBERS", 2):
            with self.assertRaisesRegex(ValueError, "limit|many"):
                self.install(archive_bytes(entries))
        with patch.object(self.api, "MAX_UNPACKED_BYTES", 1), patch.object(self.api, "MAX_MEMBERS", 1):
            with self.assertRaisesRegex(ValueError, "limit|large"):
                self.install(gzip.compress(b"\0" * 12000, mtime=0))

    def test_concurrent_installers_publish_one_complete_cache(self):
        payload = archive_bytes()
        with ThreadPoolExecutor(max_workers=8) as pool:
            installed = list(pool.map(lambda _: self.install(payload), range(16)))
        self.assertEqual(len({item.root for item in installed}), 1)
        self.assertEqual((installed[0].root / "entry.py").read_text(), "def answer():\n    return 42\n")
        self.assertEqual(len(list((self.root / "cache").iterdir())), 1)

    def test_install_does_not_execute_code_and_explicit_import_works(self):
        marker = self.root / "executed"
        source = (
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\ndef answer():\n    return 42\n"
        ).encode()
        installed = self.install(
            archive_bytes([("entry.py", source, tarfile.REGTYPE)]), prefix=self.prefix + ".generated"
        )
        self.assertFalse(marker.exists())
        self.api.activate_startup_artifacts([installed])
        self.assertFalse(marker.exists())
        entry, qualname = (installed.module_prefix + ".entry:answer").split(":")
        self.assertEqual(getattr(importlib.import_module(entry), qualname)(), 42)
        self.assertTrue(marker.exists())
        self.api.activate_startup_artifacts([installed])
        self.assertEqual(
            self.install(
                archive_bytes([("entry.py", source, tarfile.REGTYPE)]), prefix=self.prefix + ".generated"
            ).root,
            installed.root,
        )
        self.assertFalse(list(installed.root.rglob("*.pyc")))

    def test_activation_rejects_prefix_collision(self):
        first = self.install()
        second = self.install(archive_bytes([("entry.py", b"other = True", tarfile.REGTYPE)]))
        self.api.activate_startup_artifacts([first])
        with self.assertRaisesRegex(ValueError, "prefix|bound"):
            self.api.activate_startup_artifacts([second])

    def test_namespace_siblings_packages_and_relative_imports(self):
        payload = archive_bytes(
            [
                ("package/__init__.py", b"from .entry import answer\n", tarfile.REGTYPE),
                ("package/entry.py", b"from ..shared import VALUE\ndef answer():\n    return VALUE\n", tarfile.REGTYPE),
                ("shared.py", b"VALUE = 42\n", tarfile.REGTYPE),
            ]
        )
        first = self.install(payload, self.prefix + ".first")
        second = self.install(payload, self.prefix + ".second")
        self.api.activate_startup_artifacts([first, second])
        for artifact in (first, second):
            package = importlib.import_module(artifact.module_prefix + ".package")
            self.assertEqual(package.answer(), 42)
            self.assertEqual(package.__package__, artifact.module_prefix + ".package")
        self.assertFalse(list(first.root.rglob("*.pyc")))
        self.assertEqual(self.install(payload, self.prefix + ".third").root, first.root)

    def test_bytecode_only_artifact_cannot_bypass_source_loader(self):
        marker = self.root / "bytecode_executed"
        source = self.root / "bytecode_source.py"
        source.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n", encoding="utf-8")
        compiled = self.root / "entry.pyc"
        py_compile.compile(str(source), cfile=str(compiled), doraise=True)
        installed = self.install(archive_bytes([("entry.pyc", compiled.read_bytes(), tarfile.REGTYPE)]))
        self.api.activate_startup_artifacts([installed])
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module(self.prefix + ".entry")
        self.assertFalse(marker.exists())

    def test_unrelated_module_import_uses_normal_import_machinery(self):
        self.api.activate_startup_artifacts([self.install()])
        name = self.prefix + "_unrelated"
        (self.root / (name + ".py")).write_text("VALUE = 123\n", encoding="utf-8")
        self.addCleanup(lambda: sys.modules.pop(name, None))
        with patch.object(sys, "path", [str(self.root)] + sys.path):
            self.assertEqual(importlib.import_module(name).VALUE, 123)


if __name__ == "__main__":
    unittest.main()
