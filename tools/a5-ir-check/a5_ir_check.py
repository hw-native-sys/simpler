#!/usr/bin/env python3
"""Scan A5 device code for the compiler's stack-realignment miscompile.

The device compiler mis-handles a stack object with a large alignment when its
address escapes into an outlined call: one alloca ends up with two runtime
addresses a few bytes apart, so a predicate can read a shifted object and report
an error that never happened. The source looks correct, the build is clean and
unit tests pass; only an A5 run shows it.

This tool finds that configuration without a device. Point it at a Ninja build
directory: it harvests the device compile lines with `ninja -t commands` (AICore
targets are add_custom_command, so they never reach a compile_commands.json),
re-emits each TU as LLVM bitcode with -flto, and flags every stack object with
align >= 16 whose address crosses a call that is not alwaysinline.

Why -flto: ccec rejects -emit-llvm and every -mllvm -print-* option on the
device side, but LTO must consume bitcode, so `-c -flto` writes valid device
bitcode (target triple hiipu64-hisilicon-cce). A newer llvm-dis reads it.

Usage:
    a5_ir_check.py <build-dir>

Escapes are followed through getelementptr/bitcast/addrspacecast only: a value
routed through phi or select is not traced, so a hit is evidence and a clean run
is not proof.

Exit status: 0 clean, 1 findings, 2 usage/tooling error or an incomplete scan (a
TU that failed to compile or disassemble). The flags exist for unusual setups
(A/B fixtures, vendor-matched tools); the defaults are what a normal review
needs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- IR analysis

DEFINE = re.compile(r"^define\b[^\n]*?@([\w.$]+)\(")
DECLARE = re.compile(r"^declare\b[^\n]*?@([\w.$]+)\(")
ATTR_SUFFIX = re.compile(r"#(\d+)\s*\{\s*$")
ATTR_GROUP = re.compile(r"^attributes #(\d+) = \{")
ALLOCA = re.compile(r"^\s*(%[\w.$]+) = alloca\b.*?align (\d+)")
DERIVE = re.compile(r"^\s*(%[\w.$]+) = (?:getelementptr|bitcast|addrspacecast)\b(.*)$")
VALUE = re.compile(r"%[\w.$]+")
# %struct.X / %class.X / %"quoted" are type names, not values; a GEP spells its
# source type first, so taking the first %-token would capture the type.
TYPE_NAME = re.compile(r'^%(?:struct|class|union|enum)\.|^%"')
CALL = re.compile(r"^\s*(?:%[\w.$]+ = )?(?:tail\s+)?(?:call|invoke)\b.*?@([\w.$]+)\((.*)\)")
# A call through a function pointer names no @callee. `[^@\n]*?` keeps this from
# also matching direct calls, and the callee is unknown, so it can never be
# proven alwaysinline -- such escapes are reported conservatively.
CALL_INDIRECT = re.compile(r"^\s*(?:%[\w.$]+ = )?(?:tail\s+)?(?:call|invoke)\b[^@\n]*?(%[\w.$]+)\s*\((.*)\)")


@dataclass
class Finding:
    function: str
    register: str
    align: int
    callee: str

    def __str__(self) -> str:
        return f"{self.function}: {self.register} (align {self.align}) escapes to {self.callee}"


def _strip_comments(text: str) -> str:
    return "\n".join("" if line.lstrip().startswith(";") else line.split(";", 1)[0] for line in text.splitlines())


def _attribute_groups(text: str) -> dict[str, str]:
    groups = {}
    for line in text.splitlines():
        m = ATTR_GROUP.match(line)
        if m:
            groups[m.group(1)] = line
    return groups


def _symbol_attributes(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = DEFINE.match(line) or DECLARE.match(line)
        if not m:
            continue
        a = ATTR_SUFFIX.search(line)
        if a:
            out[m.group(1)] = a.group(1)
    return out


def _functions(text: str):
    current: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        m = DEFINE.match(line)
        if m:
            current = (m.group(1), [])
            continue
        if current is None:
            continue
        if line.startswith("}"):
            yield current
            current = None
        else:
            current[1].append(line)


def _derived_base(rest: str) -> str | None:
    """First value operand of a GEP/bitcast, skipping the source type name."""
    for m in VALUE.finditer(rest):
        if not TYPE_NAME.match(m.group(0)):
            return m.group(0)
    return None


def _derive_closure(body: list[str], roots: set[str]) -> dict[str, str]:
    """Map every tracked value to the alloca it was derived from.

    Values reached through phi/select are deliberately not tracked: the call
    operand is matched literally, so an escape routed that way is missed and the
    caller must not read a clean run as proof.
    """
    alias = {root: root for root in roots}
    derived = [(m, _derived_base(m.group(2))) for m in (DERIVE.match(line) for line in body) if m]
    for _ in range(8):
        grown = False
        for m, base in derived:
            result = m.group(1)
            if base in alias and result not in alias:
                alias[result] = alias[base]
                grown = True
        if not grown:
            break
    return alias


def scan_ir(text: str, min_align: int) -> list[Finding]:
    text = _strip_comments(text)
    groups = _attribute_groups(text)
    symbols = _symbol_attributes(text)
    findings: list[Finding] = []

    for name, body in _functions(text):
        escaped = {m.group(1): int(m.group(2)) for m in (ALLOCA.match(line) for line in body) if m}
        escaped = {reg: align for reg, align in escaped.items() if align >= min_align}
        if not escaped:
            continue

        aliases = _derive_closure(body, set(escaped))
        for line in body:
            m = CALL.match(line)
            if m:
                callee, operands = m.group(1), m.group(2)
                if callee.startswith("llvm."):
                    continue  # lifetime / memcpy take the address without escaping it
                if "alwaysinline" in groups.get(symbols.get(callee, ""), ""):
                    continue
            else:
                m = CALL_INDIRECT.match(line)
                if not m:
                    continue
                callee, operands = "<indirect>", m.group(2)
            # %x must not match %x.i / %x.addr: the suffix set is the whole
            # identifier character class, not \b (which stops at '.').
            hit = {root for reg, root in aliases.items() if re.search(re.escape(reg) + r"(?![\w.$])", operands)}
            for root in sorted(hit):
                findings.append(Finding(name, root, escaped[root], callee))
    return findings


# ------------------------------------------------------------ build harvesting


@dataclass
class Entry:
    directory: str
    file: str
    command: list[str]

    def text(self) -> str:
        return shlex.join(self.command)


A5_DEVICE = re.compile(r"--cce-aicore-arch=dav-c310-")
A5_COMPILE = re.compile(r"(?:-x cce\b|--cce-aicore-only)")


def is_a5_device(cmd: str) -> bool:
    # A2/A3 lines also carry --cce-aicore-only; the arch is what makes it A5.
    return bool(A5_DEVICE.search(cmd)) and bool(A5_COMPILE.search(cmd))


def _parse_line(line: str, build_dir: str) -> Entry | None:
    """One harvested command line -> Entry, keeping only the compile segment."""
    line = line.strip()
    if not line or not is_a5_device(line):
        return None
    # Both generators print "cd DIR && <compile> && <cmake depfile fixup>".
    directory, compile_seg = build_dir, line
    for seg in line.split("&&"):
        seg = seg.strip()
        if seg.startswith("cd "):
            directory = seg[3:].strip()
        elif re.search(r"/(ccec|bisheng|clang\+\+|clang)$", seg.split()[0] if seg.split() else ""):
            compile_seg = seg
    parts = shlex.split(compile_seg)
    src = next((p for p in parts if p.endswith((".cpp", ".cce", ".c"))), None)
    return Entry(directory=directory, file=src, command=parts) if src else None


def load_ninja(build_dir: str) -> list[Entry]:
    """Device TUs are add_custom_command outputs: only the generator knows their command lines."""
    out = subprocess.run(["ninja", "-t", "commands"], cwd=build_dir, capture_output=True, text=True, check=True).stdout
    entries: list[Entry] = []
    for line in out.splitlines():
        entry = _parse_line(line, build_dir)
        if entry:
            entries.append(entry)
    return entries


def load_make(build_dir: str) -> list[Entry]:
    """CMake's Makefile generator writes each target's verbatim commands into build.make."""
    entries: list[Entry] = []
    for makefile in sorted(Path(build_dir).rglob("CMakeFiles/*/build.make")):
        for line in makefile.read_text(errors="replace").splitlines():
            entry = _parse_line(line, build_dir)
            if entry:
                entries.append(entry)
    return entries


def load_entries(build_dir: str) -> list[Entry]:
    """Pick the harvest that matches the generator, so callers pass only the build dir."""
    if (Path(build_dir) / "build.ninja").exists():
        return load_ninja(build_dir)
    if any(True for _ in Path(build_dir).rglob("CMakeFiles/*/build.make")):
        return load_make(build_dir)
    raise FileNotFoundError(f"{build_dir}: neither build.ninja nor CMakeFiles/*/build.make")


# ---------------------------------------------------------------- IR emission


@dataclass
class Job:
    entry: Entry
    bitcode: Path
    text: Path
    ok: bool = True
    error: str = ""
    findings: list[Finding] = field(default_factory=list)


DEBUG_FLAG = re.compile(r"^-g($|\d|line|dwarf|split|column|pubnames|z|cold)")


def ir_command(
    entry: Entry,
    bitcode: Path,
    opt_level: str | None = None,
    keep_debug: bool = False,
    extra: list[str] | None = None,
    ccec: str | None = None,
) -> list[str]:
    """Rewrite a device compile line into a bitcode emit."""
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(entry.command):
        if skip_next:
            skip_next = False
            continue
        if tok in ("-o", "--output", "-MF", "-MT", "-MQ"):
            skip_next = True
            continue
        if tok in ("-MD", "-MMD", "-MJ"):
            continue  # dependency emission is an extra output, not IR
        if tok.startswith(("-MF", "-MT", "-MQ", "-MJ")) and len(tok) > 3:
            continue
        if tok.startswith("-o") and len(tok) > 2:
            continue
        if tok in ("-flto", "-flto=thin", "-fembed-bitcode", "-fembed-bitcode=all"):
            continue
        if opt_level is not None and re.fullmatch(r"-O\d?|-[Oo]s|-[Oo]z|-[Oo]fast", tok):
            continue
        if not keep_debug and DEBUG_FLAG.match(tok):
            continue  # DWARF for these TUs is huge and is not needed for escape analysis
        out.append(tok)
    if opt_level is not None:
        out.append(f"-O{opt_level}")
    if not keep_debug:
        out.append("-g0")
    if "-flto" not in out:
        out.append("-flto")
    # -c is what makes the driver write bitcode rather than link; keep it if present.
    if extra:
        out.extend(extra)
    out.extend(["-o", str(bitcode)])
    if ccec and out:
        out[0] = ccec  # explicit override wins over the harvested compiler
    return out


def find_tool(name: str, override: str | None) -> str | None:
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for root in (os.environ.get("ASCEND_HOME_PATH"), "/usr/local/Ascend/cann-9.3.0", "/usr/local/Ascend/cann-9.0.0"):
        if not root:
            continue
        cand = Path(root) / "ccec_compiler" / "bin" / name
        if cand.exists():
            return str(cand)
    return None


def run_job(
    job: Job,
    ccec: str | None,
    llvm_dis: str,
    min_align: int,
    keep_ir: bool,
    opt_level: str | None = None,
    keep_debug: bool = False,
    extra: list[str] | None = None,
) -> Job:
    cmd = ir_command(job.entry, job.bitcode, opt_level, keep_debug, extra, ccec)
    try:
        proc = subprocess.run(cmd, cwd=job.entry.directory, capture_output=True, text=True, check=False)
    except OSError as exc:
        job.ok, job.error = False, str(exc)
        return job
    if proc.returncode != 0 or not job.bitcode.exists():
        job.ok = False
        noise = (proc.stderr or proc.stdout).strip()
        job.error = noise.splitlines()[-1] if noise else "no bitcode produced"
        return job

    dis = subprocess.run([llvm_dis, str(job.bitcode), "-o", str(job.text)], capture_output=True, text=True, check=False)
    if dis.returncode != 0 or not job.text.exists():
        job.ok = False
        job.error = f"llvm-dis: {dis.stderr.strip().splitlines()[-1:] or 'failed'}"
        return job

    job.findings = scan_ir(job.text.read_text(errors="replace"), min_align)
    if not keep_ir:
        job.bitcode.unlink(missing_ok=True)
        job.text.unlink(missing_ok=True)
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "build_dir",
        nargs="+",
        metavar="BUILD_DIR",
        help="Ninja build directory holding the A5 device targets (`ninja -t commands` is the source)",
    )
    ap.add_argument("--min-align", type=int, default=16)
    ap.add_argument(
        "--opt-level",
        choices=["0", "1", "2", "3"],
        default=None,
        help="override -O for the scan build (use 1 to match the A5 dev profile)",
    )
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--ir-dir", default=None, help="where to keep bitcode/IR (default: temp dir)")
    ap.add_argument("--keep-ir", action="store_true")
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra flag for the scan build, e.g. -DSIMPLER_AICORE_DEV_OUTLINE=1",
    )
    ap.add_argument(
        "--keep-debug",
        action="store_true",
        help="keep -g in the scan build (slow, and the IR can be too large for llvm-dis)",
    )
    ap.add_argument("--ccec", default=None, help="compiler override (default: the one in the harvested command)")
    ap.add_argument("--llvm-dis", default=None)
    ap.add_argument("--list", action="store_true", help="only list the device TUs that would be scanned")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    entries: list[Entry] = []
    for d in args.build_dir:
        try:
            entries.extend(load_entries(d))
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"cannot harvest {d}: {exc}", file=sys.stderr)
            return 2

    device = [e for e in entries if is_a5_device(e.text())]
    if args.verbose or args.list:
        print(f"{len(entries)} device compile lines, {len(device)} A5 device TUs")
        for e in device:
            print(f"  {e.file}")
        if args.list:
            return 0
    if not device:
        print("no A5 device translation units found", file=sys.stderr)
        return 2

    # The harvested line already names its compiler, so only an explicit
    # override is resolved here; requiring a discoverable ccec would reject a
    # valid bisheng line for no reason.
    ccec = args.ccec
    llvm_dis = find_tool("llvm-dis", args.llvm_dis)
    if not llvm_dis:
        print("missing tool: llvm-dis (set --llvm-dis)", file=sys.stderr)
        return 2

    ir_dir = (Path(args.ir_dir).expanduser() if args.ir_dir else Path(tempfile.mkdtemp(prefix="a5ir-"))).resolve()
    ir_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for e in device:
        key = hashlib.sha256((e.directory + "\0" + e.text()).encode()).hexdigest()[:16]
        name = Path(e.file).stem
        jobs.append(Job(entry=e, bitcode=ir_dir / f"{name}-{key}.bc", text=ir_dir / f"{name}-{key}.ll"))

    def _one(job: Job) -> Job:
        # One bad TU must not take the whole run down: an uncaught exception
        # would surface as exit 1, which the caller reads as "findings".
        try:
            return run_job(
                job,
                ccec,
                llvm_dis,
                args.min_align,
                args.keep_ir,
                args.opt_level,
                args.keep_debug,
                args.extra_arg,
            )
        except Exception as exc:  # noqa: BLE001 - report, never abort the batch
            job.ok, job.error = False, f"{type(exc).__name__}: {exc}"
            return job

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_one, jobs))

    total = 0
    for job in results:
        label = Path(job.entry.file).name
        if not job.ok:
            print(f"SKIP {label}: {job.error}")
            continue
        if job.findings:
            for f in job.findings:
                print(f"HIT  {label}  {f}")
            total += len(job.findings)
        elif args.verbose:
            print(f"ok   {label}")

    scanned = sum(1 for j in results if j.ok)
    skipped = len(results) - scanned
    print(f"\n{total} finding(s) across {scanned}/{len(results)} device TU(s)")
    if total:
        return 1
    if skipped:
        print(f"incomplete: {skipped} TU(s) skipped, so this is not a clean result", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
