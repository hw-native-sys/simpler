# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO-ISA dependency management: resolve the pinned managed checkout.

``pto_isa.pin`` is the single source of truth for the PTO-ISA revision.
``ensure_pto_isa_root()`` always manages ``PROJECT_ROOT/build/pto-isa``:

1. Read the required commit from ``pto_isa.pin``.
2. Clone the managed checkout over HTTPS if it is missing.
3. Checkout/reset the managed checkout to the pinned commit.
4. Verify HEAD exactly matches the pin before returning.

Lock file under build/ serializes concurrent clones from parallel processes.
"""

import fcntl
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .environment import PROJECT_ROOT

logger = logging.getLogger(__name__)

_PTO_ISA_HTTPS = "https://github.com/hw-native-sys/pto-isa.git"
_PTO_ISA_PIN_RE = re.compile(r"^[0-9a-fA-F]{40}$")
PTO_ISA_PIN_FILE = "pto_isa.pin"
PTO_ISA_BUILD_METADATA = "pto_isa_build.json"


def read_pto_isa_pin(pin_path: Optional[Path] = None) -> str:
    """Read and validate the repository PTO-ISA pin."""
    path = pin_path or (PROJECT_ROOT / PTO_ISA_PIN_FILE)
    try:
        value = path.read_text().strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"PTO-ISA pin not found at {path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to read PTO-ISA pin at {path}: {e}") from e

    if not value:
        raise RuntimeError(f"Invalid PTO-ISA pin at {path}: expected a 40-character hex SHA, got an empty file")
    if not _PTO_ISA_PIN_RE.fullmatch(value):
        raise RuntimeError(f"Invalid PTO-ISA pin at {path}: expected a 40-character hex SHA, got {value!r}")
    return value.lower()


def get_pto_isa_head(pto_isa_root: str) -> str:
    """Return the full git HEAD SHA for a PTO-ISA checkout, or empty if unknown."""
    try:
        result = _run_git_resilient(["rev-parse", "HEAD"], cwd=Path(pto_isa_root), timeout=5)
        return result.stdout.strip().lower() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def pto_isa_build_metadata_path(lib_dir: Path) -> Path:
    """Return the build metadata path under build/lib."""
    return lib_dir / PTO_ISA_BUILD_METADATA


def pto_isa_runtime_artifact_key(arch: str, variant: str, runtime_name: str) -> str:
    """Return the metadata key for one runtime artifact set."""
    return f"{arch}/{variant}/{runtime_name}"


def _metadata_commit(payload: dict) -> str:
    return (
        str(
            payload.get("required_commit_from_pin")
            or payload.get("actual_checkout_commit")
            or payload.get("pto_isa_commit", "")
        )
        .strip()
        .lower()
    )


def _metadata_entry(required_commit: str, actual_commit: str, pto_isa_root: str) -> dict:
    return {
        "required_commit_from_pin": required_commit,
        "actual_checkout_commit": actual_commit,
        "pin_file": str((PROJECT_ROOT / PTO_ISA_PIN_FILE).resolve()),
        "checkout_path": str(Path(pto_isa_root).resolve()),
    }


def write_pto_isa_build_metadata(lib_dir: Path, pto_isa_root: str, runtime_keys: Iterable[str] = ()) -> None:
    """Record the pinned PTO-ISA revision used to build runtime binaries."""
    required_commit = read_pto_isa_pin()
    actual_commit = get_pto_isa_head(pto_isa_root)
    if not actual_commit:
        raise RuntimeError(
            "Cannot record PTO-ISA build revision: "
            f"{pto_isa_root} is not a git checkout or git HEAD is unavailable. "
            "Building a2a3 onboard runtimes requires the managed build/pto-isa checkout."
        )
    if actual_commit != required_commit:
        raise RuntimeError(
            "PTO-ISA checkout mismatch while recording runtime build metadata: "
            f"pto_isa.pin requires {required_commit}, but {pto_isa_root} is at {actual_commit}."
        )

    keys = [runtime_keys] if isinstance(runtime_keys, str) else sorted(dict.fromkeys(runtime_keys))
    entry = _metadata_entry(required_commit, actual_commit, pto_isa_root)
    lib_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lib_dir / ".pto_isa_build.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing = read_pto_isa_build_metadata(lib_dir) or {}
        runtime_artifacts = {}
        if existing.get("schema_version") == 3 and isinstance(existing.get("runtime_artifacts"), dict):
            runtime_artifacts.update(existing["runtime_artifacts"])
        for key in keys:
            runtime_artifacts[key] = dict(entry)

        metadata = {
            "schema_version": 3,
            "required_commit_from_pin": required_commit,
            "actual_checkout_commit": actual_commit,
            "pin_file": str((PROJECT_ROOT / PTO_ISA_PIN_FILE).resolve()),
            "checkout_path": str(Path(pto_isa_root).resolve()),
            "runtime_artifacts": runtime_artifacts,
        }
        metadata_path = pto_isa_build_metadata_path(lib_dir)
        tmp_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(metadata_path)


def read_pto_isa_build_metadata(lib_dir: Path) -> Optional[dict]:
    """Read installed runtime PTO-ISA metadata, if present."""
    metadata_path = pto_isa_build_metadata_path(lib_dir)
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: expected JSON object")
    return payload


def validate_runtime_pto_isa_current_pin(lib_dir: Path, runtime_key: Optional[str] = None) -> None:
    """Raise when pre-built runtime binaries do not match the current pin."""
    metadata = read_pto_isa_build_metadata(lib_dir)
    if metadata is None:
        return

    required_commit = read_pto_isa_pin()
    metadata_path = pto_isa_build_metadata_path(lib_dir)
    if metadata.get("schema_version") == 3 and runtime_key is not None:
        artifacts = metadata.get("runtime_artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: expected runtime_artifacts object")
        artifact = artifacts.get(runtime_key)
        if artifact is None:
            raise RuntimeError(
                "Stale PTO-ISA runtime binaries: current pto_isa.pin requires "
                f"{required_commit}, but {metadata_path} has no entry for runtime {runtime_key!r}.\n"
                "Reinstall simpler or rebuild this runtime so build/lib matches pto_isa.pin."
            )
        if not isinstance(artifact, dict):
            raise RuntimeError(
                f"Invalid PTO-ISA build metadata at {metadata_path}: "
                f"runtime_artifacts[{runtime_key!r}] must be a JSON object"
            )
        build_commit = _metadata_commit(artifact)
    else:
        build_commit = _metadata_commit(metadata)
    if not build_commit or build_commit == required_commit:
        return

    raise RuntimeError(
        "Stale PTO-ISA runtime binaries: current pto_isa.pin requires "
        f"{required_commit}, but installed runtimes were built for {build_commit}.\n"
        f"Build metadata: {metadata_path}\n"
        "Reinstall simpler or rebuild runtimes so build/lib matches pto_isa.pin."
    )


def get_pto_isa_clone_path() -> Path:
    """Managed auto-clone target for PTO-ISA, anchored to PROJECT_ROOT."""
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


def _run_git(
    args: list,
    cwd: Optional[Path] = None,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a git subcommand and capture stdout/stderr as text."""
    return subprocess.run(
        ["git"] + args,
        check=check,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )


def _is_dubious_ownership_error(stderr: str) -> bool:
    return "detected dubious ownership" in stderr and "safe.directory" in stderr


def _run_git_with_safe_directory(
    args: list,
    cwd: Path,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return _run_git(["-c", f"safe.directory={cwd.resolve()}", *args], cwd=cwd, timeout=timeout, check=check)


def _run_git_resilient(
    args: list,
    cwd: Path,
    timeout: int = 30,
    check: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run git, retrying dubious-ownership failures with per-command safe.directory."""
    result = _run_git(args, cwd=cwd, timeout=timeout, check=False)
    if result.returncode != 0 and _is_dubious_ownership_error(result.stderr):
        if verbose:
            logger.info(f"Using pto-isa safe.directory for {cwd}")
        result = _run_git_with_safe_directory(args, cwd=cwd, timeout=timeout, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ["git", *args], result.stdout, result.stderr)
    return result


def _discard_incomplete_clone(target: Path, verbose: bool) -> None:
    """Remove `target` if it exists but is not a usable clone."""
    if not (target.exists() or target.is_symlink()) or _is_cloned(target):
        return
    if verbose:
        logger.warning(f"Removing incomplete pto-isa clone at {target}")
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)


def _clone(target: Path, verbose: bool) -> bool:
    """Clone PTO-ISA to `target` over HTTPS. Returns True on success."""
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

    _discard_incomplete_clone(target, verbose)
    logger.info(f"Cloning pto-isa to {target} over HTTPS (first run, may take up to a minute)...")

    try:
        result = _run_git(["clone", _PTO_ISA_HTTPS, str(target)], timeout=300)
        if result.returncode != 0:
            if verbose:
                logger.warning(f"Failed to clone pto-isa:\n{result.stderr}")
            _discard_incomplete_clone(target, verbose)
            return False
        if verbose:
            logger.info(f"pto-isa cloned successfully: {target}")
        return True
    except subprocess.TimeoutExpired:
        if verbose:
            logger.warning("Clone operation timed out")
        _discard_incomplete_clone(target, verbose)
        return False
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.warning(f"Failed to clone pto-isa: {e}")
        _discard_incomplete_clone(target, verbose)
        return False


def checkout_pto_isa_commit(clone_path: Path, commit: str, verbose: bool = False) -> bool:
    """Checkout/reset the managed clone to `commit`. Return False on failure."""
    try:
        _run_git_resilient(["reset", "--hard"], cwd=clone_path, timeout=30, check=True, verbose=verbose)
        _run_git_resilient(["clean", "-fdx"], cwd=clone_path, timeout=30, check=True, verbose=verbose)
        result = _run_git_resilient(
            ["checkout", "--force", "--detach", commit], cwd=clone_path, timeout=30, verbose=verbose
        )
        if result.returncode != 0:
            if verbose:
                logger.info(f"pto-isa commit {commit} missing locally, fetching origin...")
            _run_git_resilient(["fetch", "origin"], cwd=clone_path, timeout=120, check=True, verbose=verbose)
            _run_git_resilient(
                ["checkout", "--force", "--detach", commit], cwd=clone_path, timeout=30, check=True, verbose=verbose
            )
        _run_git_resilient(["reset", "--hard", commit], cwd=clone_path, timeout=30, check=True, verbose=verbose)
        _run_git_resilient(["clean", "-fdx"], cwd=clone_path, timeout=30, check=True, verbose=verbose)
        actual = get_pto_isa_head(str(clone_path))
        if actual != commit:
            logger.warning(f"pto-isa checkout verification failed: expected {commit}, got {actual or '<unknown>'}")
            return False
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to checkout pto-isa commit {commit}: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking out pto-isa commit {commit}: {e}")
        return False


def ensure_pto_isa_root(verbose: bool = False) -> str:
    """Resolve the pinned managed PTO-ISA checkout. Return absolute path."""
    required_commit = read_pto_isa_pin()
    clone_path = get_pto_isa_clone_path()
    lock_path = clone_path.parent / ".pto-isa.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        resolved = _ensure_locked(clone_path, required_commit=required_commit, verbose=verbose)

    if resolved is None:
        raise OSError(
            f"PTO-ISA not available.\n"
            f"  The managed checkout must live at {clone_path} and match {PROJECT_ROOT / PTO_ISA_PIN_FILE}.\n"
            f"  If auto-clone failed, manually run:\n"
            f"    git clone {_PTO_ISA_HTTPS} {clone_path}"
        )
    return resolved


def _ensure_locked(clone_path: Path, required_commit: str, verbose: bool) -> Optional[str]:
    """Inner logic executed while holding the file lock."""
    if not _is_cloned(clone_path):
        if not _clone(clone_path, verbose=verbose):
            if not _is_cloned(clone_path):
                return None
            if verbose:
                logger.info("pto-isa already cloned by another process")

    if not checkout_pto_isa_commit(clone_path, required_commit, verbose=verbose):
        return None

    if not _is_cloned(clone_path):
        if verbose:
            logger.warning(f"pto-isa path exists but missing include directory: {clone_path / 'include'}")
        return None

    actual_commit = get_pto_isa_head(str(clone_path))
    if actual_commit != required_commit:
        if verbose:
            logger.warning(f"pto-isa HEAD mismatch: expected {required_commit}, got {actual_commit or '<unknown>'}")
        return None

    return str(clone_path.resolve())
