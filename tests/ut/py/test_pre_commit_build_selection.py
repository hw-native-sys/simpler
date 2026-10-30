# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]
WORKFLOW = REPO_ROOT / ".github/workflows/_pre-commit.yml"
PATH_HELPER = REPO_ROOT / "tests/lint/clang_tidy_paths.py"
CLANG_TIDY = REPO_ROOT / "tests/lint/clang_tidy.py"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def _load_path_helper():
    spec = importlib.util.spec_from_file_location("clang_tidy_paths", PATH_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATH_HELPER_MODULE = _load_path_helper()


def _load_clang_tidy_module(monkeypatch):
    monkeypatch.syspath_prepend(str(CLANG_TIDY.parent))
    spec = importlib.util.spec_from_file_location("clang_tidy_under_test", CLANG_TIDY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["run"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == "Select lint build target")


def _commit(repo: Path, relative_path: str, contents: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    subprocess.run(["git", "add", relative_path], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", relative_path],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _run_selection(repo: Path, base: str, head: str, output: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_WORKSPACE": str(REPO_ROOT),
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _selection_script()],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    diagnostics = result.stdout + result.stderr
    assert result.returncode == 0, diagnostics
    assert output.is_file(), diagnostics or "selection script did not write GITHUB_OUTPUT"
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def test_selection_script_delegates_path_matching_to_the_shared_helper() -> None:
    script = _selection_script()

    assert 'python "$GITHUB_WORKSPACE/tests/lint/clang_tidy_paths.py" --needs-build --null' in script
    assert 'case "$path"' not in script


def test_clang_tidy_hook_delegates_path_matching_to_the_shared_helper() -> None:
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text())
    clang_tidy_hook = next(
        hook
        for repository in config["repos"]
        if repository["repo"] == "local"
        for hook in repository["hooks"]
        if hook["id"] == "clang-tidy"
    )

    assert "types_or" not in clang_tidy_hook
    assert "exclude" not in clang_tidy_hook
    assert "from clang_tidy_paths import should_run_clang_tidy" in CLANG_TIDY.read_text()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/common/api.cpp", True),
        ("3rdparty/dependency.cpp", False),
        ("python/bindings/module.cpp", False),
        ("examples/a2a3/demo/kernels/aic/kernel.cpp", False),
        ("src/a2a3/runtime/demo/aicore/kernel.cpp", False),
        (str(REPO_ROOT / "python/bindings/module.cpp"), False),
        (str(REPO_ROOT / "examples/a2a3/demo/kernels/aic/kernel.cpp"), False),
        (str(REPO_ROOT / "src/common/api.cpp"), True),
        ("Python/Bindings/module.cpp", True),
        ("src/a2a3/runtime/demo/Kernels/kernel.cpp", True),
    ],
)
def test_clang_tidy_path_selection_has_one_case_sensitive_contract(path: str, expected: bool) -> None:
    assert PATH_HELPER_MODULE.should_run_clang_tidy(path) is expected


def test_missing_compile_driver_falls_back_to_the_cxx_on_path(monkeypatch, tmp_path: Path) -> None:
    module = _load_clang_tidy_module(monkeypatch)
    missing = tmp_path / "c++"
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/c++" if name == "c++" else None)

    rewritten = module._rewrite_argv([str(missing), "-c", "file.cpp"])

    assert rewritten[0] == "/usr/bin/c++"


def test_newer_clang_tidy_drops_the_scoped_lock_check(monkeypatch) -> None:
    module = _load_clang_tidy_module(monkeypatch)

    def fake_run(argv, **kwargs):
        assert argv[:3] == ["clang-tidy", "--checks=modernize-use-scoped-lock", "--list-checks"]
        return subprocess.CompletedProcess(argv, 0, stdout="    modernize-use-scoped-lock\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._scoped_lock_override() == ["--checks=-modernize-use-scoped-lock"]


def test_clang_tidy_18_keeps_its_configured_checks(monkeypatch) -> None:
    module = _load_clang_tidy_module(monkeypatch)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="Enabled checks:\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._scoped_lock_override() == []


def test_clang_tidy_header_filter_ignores_third_party_include_trees() -> None:
    pattern = re.compile(yaml.safe_load((REPO_ROOT / ".clang-tidy").read_text())["HeaderFilterRegex"])
    third_party = "/home/runner/work/simpler/simpler/build/pto-isa/include/pto/comm/async/urma/urma_types.hpp"
    project = "/home/runner/work/simpler/simpler/src/common/platform/include/common/host_api.h"

    assert pattern.search(third_party) is None
    assert pattern.search(project) is not None


def test_libstdcxx_paths_stay_empty_when_clang_tidy_sees_atomic(monkeypatch) -> None:
    module = _load_clang_tidy_module(monkeypatch)
    monkeypatch.setattr(module, "_tidy_needs_libstdcxx_isystem", lambda: False)

    assert module._libstdcxx_isystem_args() == []


def test_missing_stdlib_uses_host_gxx_include_paths(monkeypatch) -> None:
    module = _load_clang_tidy_module(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/c++" if name in {"c++", "g++"} else None)

    def fake_run(argv, **kwargs):
        assert argv[0] == "/usr/bin/c++"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr=(
                "#include <...> search starts here:\n"
                " /usr/include/c++/12\n"
                " /usr/include/c++/12/backward\n"
                " /usr/lib/gcc/include\n"
                "End of search list.\n"
            ),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._host_libstdcxx_isystem_args() == [
        "--extra-arg=-isystem/usr/include/c++/12",
        "--extra-arg=-isystem/usr/include/c++/12/backward",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "3rdparty/dependency.cpp",
        "python/bindings/module.cpp",
        "examples/a2a3/demo/kernels/aic/kernel.cpp",
        "src/a2a3/runtime/demo/aicore/kernel.cpp",
    ],
)
def test_clang_tidy_excluded_paths_exit_before_build_or_compile_database(monkeypatch, path: str) -> None:
    module = _load_clang_tidy_module(monkeypatch)

    def unexpected_build() -> None:
        pytest.fail("excluded path reached clang-tidy build/cache handling")

    monkeypatch.setattr(module, "_ensure_sim_cache", unexpected_build)
    monkeypatch.setattr(sys, "argv", [str(CLANG_TIDY), path])

    assert module.main() == 0


def test_self_cpu_uses_the_project_venv_for_lint() -> None:
    workflow = WORKFLOW.read_text()
    setup = "      - name: venv + deps\n"
    lint_only = "      - name: Lint-only venv\n"
    activate = "      - name: Use lint venv\n"
    run = "      - name: Run pre-commit\n"

    assert workflow.count(setup) == workflow.count(lint_only) == workflow.count(activate) == workflow.count(run) == 1
    assert workflow.index(setup) < workflow.index(lint_only) < workflow.index(activate) < workflow.index(run)
    lint_only_block = workflow.split(lint_only, 1)[1].split(activate, 1)[0]
    assert "steps.lint-build.outputs.needs_build != 'true'" in lint_only_block
    assert "run: python3 -m venv .venv" in lint_only_block
    block = workflow.split(activate, 1)[1].split(run, 1)[0]
    assert "if: inputs.setup_variant == 'self-cpu'" in block
    assert 'run: echo "$PWD/.venv/bin" >> "$GITHUB_PATH"' in block


def test_only_selected_build_paths_install_the_project() -> None:
    workflow = WORKFLOW.read_text()
    cache = "      - name: Cache pip packages\n"
    install = "      - name: Install package (for clang-tidy compile databases)\n"
    setup = "      - name: venv + deps\n"
    lint_only = "      - name: Lint-only venv\n"

    assert "pip install torch" not in workflow
    cache_block = workflow.split(cache, 1)[1].split(install, 1)[0]
    assert "inputs.setup_variant == 'self-cpu' || steps.lint-build.outputs.needs_build == 'true'" in cache_block
    install_block = workflow.split(install, 1)[1].split(setup, 1)[0]
    assert "steps.lint-build.outputs.needs_build == 'true'" in install_block
    assert "build.targets=build_package_sim" in install_block
    setup_block = workflow.split(setup, 1)[1].split(lint_only, 1)[0]
    assert 'install-torch: "false"' in setup_block


@pytest.mark.parametrize(
    ("changed_files", "expected"),
    [
        (["docs/readme.md"], {"needs_build": "false"}),
        ([".github/actions/setup-gcc-15/ubuntu-toolchain-r-test.asc"], {"needs_build": "false"}),
        ([".pre-commit-config.yaml"], {"needs_build": "true"}),
        ([".clang-tidy"], {"needs_build": "false"}),
        ([".clang-format"], {"needs_build": "false"}),
        ([".gitignore"], {"needs_build": "false"}),
        (["python/simpler/api.py"], {"needs_build": "false"}),
        (["python/simpler/api.PYI"], {"needs_build": "false"}),
        (["tools/server.wsgi"], {"needs_build": "false"}),
        (["3rdparty/tool.py"], {"needs_build": "false"}),
        (["src/common/api.cpp"], {"needs_build": "true"}),
        (["src/common/api.inl"], {"needs_build": "true"}),
        (["src/common/api.HPP"], {"needs_build": "true"}),
        (["vendor/dependency.cpp"], {"needs_build": "true"}),
        (["3rdparty/dependency.cpp"], {"needs_build": "false"}),
        (["3rdparty/dependency.CPP"], {"needs_build": "false"}),
        (["3RDPARTY/dependency.cpp"], {"needs_build": "true"}),
        (["python/bindings/module.cpp"], {"needs_build": "false"}),
        (["examples/a2a3/demo/kernels/aic/kernel.cpp"], {"needs_build": "false"}),
        (["src/a2a3/runtime/demo/aicore/kernel.cpp"], {"needs_build": "false"}),
        (["src/common/future.newcpp"], {"needs_build": "true"}),
        (["python/simpler/api.py", "include/simpler/api.h"], {"needs_build": "true"}),
    ],
)
def test_select_lint_build_target(tmp_path: Path, changed_files: list[str], expected: dict[str, str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    for index, changed_file in enumerate(changed_files):
        head = _commit(repo, changed_file, f"change {index}\n")

    actual = _run_selection(repo, base, head, tmp_path / "github-output")
    assert actual == expected


def test_deleted_source_does_not_require_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "src/common/deleted.cpp", "int deleted;\n")
    (repo / "src/common/deleted.cpp").unlink()
    subprocess.run(["git", "add", "src/common/deleted.cpp"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "delete"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "false"}


def test_special_characters_in_source_name_require_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "src/common/api name\npart.cpp", "int special_name;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_unrecognized_path_with_trailing_newline_requires_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "src/common/not-source.CPP\n", "not source\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_unrecognized_executable_script_requires_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    script = repo / "tools/run-check"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n")
    script.chmod(0o755)
    subprocess.run(["git", "add", "tools/run-check"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "script"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_selection_uses_merge_base_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    common = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "docs/readme.md", "docs\n")
    subprocess.run(["git", "checkout", "-q", "--detach", common], cwd=repo, check=True)
    base = _commit(repo, "src/common/base_only.cpp", "int base_only;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "false"}


def test_selection_falls_back_when_refs_have_no_merge_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    subprocess.run(["git", "checkout", "-q", "--orphan", "unrelated"], cwd=repo, check=True)
    (repo / "README.md").unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    head = _commit(repo, "src/common/unrelated.cpp", "int unrelated;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}
