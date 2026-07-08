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
2. If a managed checkout already exists **clean and at exactly the pin**, use it
   as-is — it already *is* the pinned ISA, so no checkout and no network.
3. Otherwise (missing, wrong revision, or dirty) obtain the pin fresh: clone
   over HTTPS (``--no-checkout`` so the default branch is never materialized)
   and force-check-out the pin.
4. Verify HEAD exactly matches the pin before returning.

Two deliberate choices:

- **Obtain the pin fresh instead of patching a dirty cache.** A checkout is
  reused only when it is provably already the pin (clean working tree with
  HEAD == pin); anything else is re-cloned rather than reset/force-checked-out
  in place, so the build never uses an ISA tree that was checked out over local
  modifications. Reusing a pristine cache keeps the common path network-free;
  the one network hop of a fresh clone is retried on transient failure.
- **Force the checkout when landing a fresh clone.** pto-isa's default branch
  carries case-duplicate doc paths (``docs/isa/TADDDEQRELU.md`` vs
  ``TAddDeqRelu.md``) that collide on a case-insensitive filesystem (macOS CI),
  leaving even a fresh working tree "modified"; a plain checkout then aborts.
  ``--no-checkout`` (never materialize the default branch) plus a forced
  checkout of the pin sidesteps that entirely.

Lock file under build/ serializes concurrent clones from parallel processes.
"""

import fcntl
import json
import logging
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .environment import PROJECT_ROOT

# A fresh clone is only needed when no usable local checkout exists; when it is,
# guard the single network hop against transient failures (e.g. GitHub
# SSL_ERROR_SYSCALL) with a few backed-off retries before giving up.
_CLONE_ATTEMPTS = 3
_CLONE_RETRY_BACKOFF_S = 2

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


def _remove_clone(target: Path, verbose: bool) -> None:
    """Unconditionally remove `target` so a fresh clone can replace it.

    Unlike ``_discard_incomplete_clone``, this removes even a *valid* clone —
    used when the managed checkout is at the wrong (or a dirty) revision and we
    re-clone at the pinned commit instead of checking out over local changes.
    Handles a directory, a plain file, or a (possibly broken) symlink.
    """
    if not (target.exists() or target.is_symlink()):
        return
    if verbose:
        logger.info(f"Removing pto-isa checkout at {target} to re-clone at the pinned commit")
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)


def _land_on_commit(clone_path: Path, commit: str, verbose: bool) -> bool:
    """Force-detach-checkout a freshly cloned tree onto `commit`. False on failure.

    `--force` is load-bearing, not defensive: pto-isa's default branch carries
    paths that differ only in case (e.g. ``docs/isa/TADDDEQRELU.md`` vs
    ``docs/isa/TAddDeqRelu.md``). On a case-insensitive filesystem (macOS CI)
    they collide onto one inode, so even a *fresh* clone's working tree reports
    them as modified, and a plain checkout to the pin aborts with "local changes
    would be overwritten." Forcing discards that pseudo-dirt and lands exactly on
    the pin. A full clone already carries every branch's history, but keep a
    fetch fallback in case the pin is not reachable from the default fetch.
    """
    try:
        result = _run_git_resilient(
            ["checkout", "--detach", "--force", commit], cwd=clone_path, timeout=30, verbose=verbose
        )
        if result.returncode != 0:
            if verbose:
                logger.info(f"pto-isa commit {commit} missing locally, fetching origin...")
            _run_git_resilient(["fetch", "origin"], cwd=clone_path, timeout=120, check=True, verbose=verbose)
            _run_git_resilient(
                ["checkout", "--detach", "--force", commit], cwd=clone_path, timeout=30, check=True, verbose=verbose
            )
        actual = get_pto_isa_head(str(clone_path))
        if actual != commit:
            logger.warning(f"pto-isa checkout verification failed: expected {commit}, got {actual or '<unknown>'}")
            return False
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check out pto-isa commit {commit}: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking out pto-isa commit {commit}: {e}")
        return False


def _is_pristine_at_commit(clone_path: Path, commit: str, verbose: bool) -> bool:
    """True iff the checkout is clean AND already at exactly `commit`.

    When both hold, the working tree already *is* the pinned ISA — git content
    is addressed by SHA, so a clean tree at HEAD == pin is byte-for-byte the pin
    — and can be used as-is with no checkout and no network. A dirty or
    wrong-revision tree returns False; the caller then obtains the pin with a
    fresh clone rather than checking out over the existing tree, so the build is
    never fed a checkout that was patched over local modifications.
    """
    if get_pto_isa_head(str(clone_path)) != commit:
        return False
    try:
        result = _run_git_resilient(["status", "--porcelain"], cwd=clone_path, timeout=30, verbose=verbose)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check pto-isa checkout cleanliness: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking pto-isa checkout cleanliness: {e}")
        return False
    return result.returncode == 0 and not result.stdout.strip()


def _clone(target: Path, commit: str, verbose: bool) -> bool:
    """Fresh-clone PTO-ISA to `target` over HTTPS and land it on `commit`.

    Any existing checkout at `target` is removed first, so this always yields a
    clean tree at exactly `commit` — the sync can never be blocked by local
    modifications in a preexisting (cached/preset) managed checkout.
    """
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

    _remove_clone(target, verbose)
    logger.info(f"Cloning pto-isa to {target} over HTTPS at {commit} (may take up to a minute)...")

    try:
        # --no-checkout: never materialize the default branch. Its working tree
        # is what carries the case-colliding docs/isa/TADDDEQRELU* paths; we only
        # ever want the pinned commit's tree, laid down by _land_on_commit.
        result = _run_git(["clone", "--no-checkout", _PTO_ISA_HTTPS, str(target)], timeout=300)
        for attempt in range(2, _CLONE_ATTEMPTS + 1):
            if result.returncode == 0:
                break
            if verbose:
                logger.warning(f"pto-isa clone attempt {attempt - 1}/{_CLONE_ATTEMPTS} failed:\n{result.stderr}")
            _remove_clone(target, verbose)  # clear the partial clone before retrying
            time.sleep(_CLONE_RETRY_BACKOFF_S * (attempt - 1))
            result = _run_git(["clone", "--no-checkout", _PTO_ISA_HTTPS, str(target)], timeout=300)
        if result.returncode != 0:
            if verbose:
                logger.warning(f"Failed to clone pto-isa after {_CLONE_ATTEMPTS} attempts:\n{result.stderr}")
            _discard_incomplete_clone(target, verbose)
            return False
        if not _land_on_commit(target, commit, verbose=verbose):
            # The clone succeeded but sits at the wrong commit (default HEAD),
            # so it looks like a valid clone and `_discard_incomplete_clone`
            # would leave it. Remove it outright so no wrong-revision checkout
            # is stranded on disk for a later run to mistake for the pin.
            _remove_clone(target, verbose)
            return False
        if verbose:
            logger.info(f"pto-isa cloned at {commit}: {target}")
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
    # Reuse an existing checkout ONLY when it is already exactly the pin: a clean
    # working tree at HEAD == pin already *is* the pinned ISA (git objects are
    # content-addressed), so use it as-is — no checkout, no network. This is the
    # warm-cache path on persistent runners.
    if _is_cloned(clone_path) and _is_pristine_at_commit(clone_path, required_commit, verbose=verbose):
        return str(clone_path.resolve())

    # Otherwise (missing, wrong revision, or dirty) obtain the pin *fresh* rather
    # than checking out over the existing tree: re-clone directly at the pin. We
    # never reset/force-checkout a dirty cache in place, so the build never uses
    # a checkout that was patched over local modifications.
    if not _clone(clone_path, required_commit, verbose=verbose):
        # A parallel process holding a separate lock may have prepared the
        # checkout concurrently; accept its result only if it landed on the pin.
        if not (_is_cloned(clone_path) and get_pto_isa_head(str(clone_path)) == required_commit):
            return None
        if verbose:
            logger.info("pto-isa prepared at the pin by another process")

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
