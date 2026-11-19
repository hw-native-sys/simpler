# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("discovery", ["python_path", "explicit_dir", "missing"])
def test_cmake_finds_nanobind_for_selected_python(tmp_path: Path, discovery: str):
    nanobind = pytest.importorskip("nanobind")
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the packaging configuration test")

    isolated = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(isolated)
    python = isolated / "bin" / "python"
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "CMAKE_PREFIX_PATH", "nanobind_DIR", "nanobind_ROOT"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["VIRTUAL_ENV"] = str(isolated)

    package_root = tmp_path / "external site"
    package_root.mkdir()
    (package_root / "nanobind").symlink_to(Path(nanobind.__file__).parent, target_is_directory=True)
    package_config = package_root / "nanobind" / "cmake"
    arguments = []
    if discovery == "python_path":
        # Python can import this external package, but it is not a CMake prefix.
        env["PYTHONPATH"] = str(package_root)
        subprocess.run([str(python), "-m", "nanobind", "--cmake_dir"], env=env, check=True, capture_output=True)
    elif discovery == "explicit_dir":
        arguments.append(f"-Dnanobind_DIR:PATH={package_config}")

    build = tmp_path / "build"
    result = subprocess.run(
        [
            cmake,
            "-S",
            str(PROJECT_ROOT),
            "-B",
            str(build),
            f"-DPython_EXECUTABLE={python}",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
            *arguments,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    if discovery == "missing":
        assert result.returncode != 0, output
        assert "Cannot locate nanobind for" in output
        assert str(python) in output
    else:
        assert result.returncode == 0, output
        cache = (build / "CMakeCache.txt").read_text()
        assert f"nanobind_DIR:PATH={package_config}" in cache
