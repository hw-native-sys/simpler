# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO-ISA dependency management: resolve or auto-clone the repo.

Single source of truth for locating / cloning / pinning the PTO-ISA repo.
Callers: root conftest (session-level pre-clone), SceneTestCase (lazy resolve
at compile time), `SceneTestCase.run_module` (pin via `-c`).

Resolution order for ensure_pto_isa_root():
  1. PTO_ISA_ROOT environment variable (if set and points to a directory)
  2. PROJECT_ROOT / build / pto-isa (auto-clone if missing)

Lock file under build/ serializes concurrent clones from parallel processes.
"""

import fcntl
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from .environment import PROJECT_ROOT

logger = logging.getLogger(__name__)

_PTO_ISA_HTTPS = "https://github.com/PTO-ISA/pto-isa.git"
_PTO_ISA_SSH = "git@github.com:PTO-ISA/pto-isa.git"


def get_pto_isa_clone_path() -> Path:
    """Default auto-clone target for PTO-ISA, anchored to PROJECT_ROOT.

    Lives under PROJECT_ROOT/build/ so each repo / worktree / venv has its own
    isolated clone (no races when multiple worktrees pin different commits).
    """
    return PROJECT_ROOT / "build" / "pto-isa"


def _is_cloned(path: Path) -> bool:
    """Return True if `path` looks like a valid PTO-ISA clone (has include/)."""
    return (path / "include").is_dir()


def _is_git_available() -> bool:
    try:
        result = subprocess.run(["git", "--version"], check=False, capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _repo_url(clone_protocol: str) -> str:
    return _PTO_ISA_HTTPS if clone_protocol == "https" else _PTO_ISA_SSH


def _run_git(
    args: list, cwd: Optional[Path] = None, timeout: int = 30, check: bool = False
) -> subprocess.CompletedProcess:
    """Run a git subcommand.

    Always captures stdout/stderr as text. `check=False` (default) returns the
    CompletedProcess for manual returncode inspection; `check=True` raises
    CalledProcessError on non-zero exit.
    """
    return subprocess.run(
        ["git"] + args,
        check=check,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )


def _clone(target: Path, commit: Optional[str], clone_protocol: str, verbose: bool) -> bool:
    """Clone PTO-ISA to `target`, optionally at `commit`. Returns True on success."""
    if not _is_git_available():
        if verbose:
            logger.warning("git command not available, cannot clone pto-isa")
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if verbose:
            logger.warning(f"Failed to create clone parent dir: {e}")
        return False

    repo_url = _repo_url(clone_protocol)
    logger.info(f"Cloning pto-isa to {target} (first run, may take up to a minute)...")

    try:
        result = _run_git(["clone", repo_url, str(target)], timeout=300)
        if result.returncode != 0:
            if verbose:
                logger.warning(f"Failed to clone pto-isa:\n{result.stderr}")
            return False

        if commit:
            result = _run_git(["checkout", commit], cwd=target, timeout=30)
            if result.returncode != 0:
                if verbose:
                    logger.warning(f"Failed to checkout pto-isa commit {commit}:\n{result.stderr}")
                return False

        if verbose:
            suffix = f" at commit {commit}" if commit else ""
            logger.info(f"pto-isa cloned successfully{suffix}: {target}")
        return True
    except subprocess.TimeoutExpired:
        if verbose:
            logger.warning("Clone operation timed out")
        return False
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.warning(f"Failed to clone pto-isa: {e}")
        return False


def checkout_pto_isa_commit(clone_path: Path, commit: str, verbose: bool = False) -> None:
    """Switch an existing clone to `commit` if it isn't already there (idempotent)."""
    try:
        result = _run_git(["rev-parse", "--short", "HEAD"], cwd=clone_path, timeout=5)
        current = result.stdout.strip() if result.returncode == 0 else ""
        if current and not commit.startswith(current) and not current.startswith(commit):
            if verbose:
                logger.info(f"pto-isa at {current}, checking out {commit}...")
            _run_git(["fetch", "origin"], cwd=clone_path, timeout=120, check=True)
            _run_git(["checkout", commit], cwd=clone_path, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to checkout pto-isa commit {commit}: {e.stderr if hasattr(e, 'stderr') else e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking out pto-isa commit {commit}: {e}")


def _update_to_latest(clone_path: Path, verbose: bool) -> None:
    """Fetch and reset existing clone to the remote default branch."""
    try:
        if verbose:
            logger.info("Updating pto-isa to latest...")
        _run_git(["fetch", "origin"], cwd=clone_path, timeout=120, check=True)
        _run_git(["reset", "--hard", "origin/HEAD"], cwd=clone_path, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to update pto-isa to latest: {e.stderr if hasattr(e, 'stderr') else e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error updating pto-isa: {e}")


def ensure_pto_isa_root(
    commit: Optional[str] = None,
    clone_protocol: str = "ssh",
    update_if_exists: bool = False,
    verbose: bool = False,
) -> str:
    """Resolve or auto-clone PTO-ISA. Return absolute path.

    Args:
        commit: if provided, check out this revision after clone/in existing clone.
        clone_protocol: "ssh" (default) or "https".
        update_if_exists: when `commit` is None and a clone already exists,
            fetch origin and reset to origin/HEAD. Conftest passes True to
            guarantee the local clone isn't stale (in particular, to ensure
            any later `-c <commit>` request resolves to a real commit rather
            than a missing object in an old clone). Lazy callers pass False
            to avoid redundant network traffic since conftest already ran.
        verbose: log progress via `logger.info` / `logger.warning`.

    Raises:
        OSError: when PTO_ISA_ROOT is unset and auto-clone fails.
    """
    env_root = os.environ.get("PTO_ISA_ROOT")
    if env_root and Path(env_root).is_dir():
        if verbose:
            logger.info(f"Using existing PTO_ISA_ROOT: {env_root}")
        return env_root

    clone_path = get_pto_isa_clone_path()
    lock_path = clone_path.parent / ".pto-isa.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        resolved = _ensure_locked(
            clone_path,
            commit=commit,
            clone_protocol=clone_protocol,
            update_if_exists=update_if_exists,
            verbose=verbose,
        )

    if resolved is None:
        raise OSError(
            f"PTO-ISA not available.\n"
            f"  Either export PTO_ISA_ROOT=/path/to/pto-isa,\n"
            f"  or manually clone to {clone_path}:\n"
            f"    git clone {_repo_url(clone_protocol)} {clone_path}"
        )
    return resolved


def _ensure_locked(
    clone_path: Path,
    commit: Optional[str],
    clone_protocol: str,
    update_if_exists: bool,
    verbose: bool,
) -> Optional[str]:
    """Inner logic executed while holding the file lock."""
    if not _is_cloned(clone_path):
        if not _clone(clone_path, commit=commit, clone_protocol=clone_protocol, verbose=verbose):
            # A parallel process may have won the race
            if not _is_cloned(clone_path):
                return None
            if verbose:
                logger.info("pto-isa already cloned by another process")
            if commit:
                checkout_pto_isa_commit(clone_path, commit, verbose=verbose)
            elif update_if_exists:
                _update_to_latest(clone_path, verbose=verbose)
    elif commit:
        checkout_pto_isa_commit(clone_path, commit, verbose=verbose)
    elif update_if_exists:
        _update_to_latest(clone_path, verbose=verbose)

    if not _is_cloned(clone_path):
        if verbose:
            logger.warning(f"pto-isa path exists but missing include directory: {clone_path / 'include'}")
        return None

    return str(clone_path.resolve())
