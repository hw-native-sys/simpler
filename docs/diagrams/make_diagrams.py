#!/usr/bin/env python3
"""Render the SIMPLER architecture diagrams as a single self-contained SVG:
  1. L0-L6 level model (the 7-layer hierarchy)
  2. Three engine components + L2 three-program model
  3. Rust-suitability map over the architecture
Run: python3 make_diagrams.py  ->  architecture.svg
Pure stdlib; no external deps. Edit here and re-run to regenerate.
"""
import html, os

W = 1180
PADDED = []  # (svg fragment, height) appended per panel

# ---- palette ----
BG = "#0e1116"; PANEL = "#161b22"; INK = "#e6edf3"; MUTE = "#8b949e"
LINE = "#30363d"
ON_DEV = "#7c4a2d"   # on-device (L0-L2) accent
HOST = "#1f4e6b"     # host/cluster accent
STRONG = "#2ea043"   # rust strong
MOD = "#9e6a1f"      # rust moderate
WEAK = "#6e3b3b"     # rust weak/no
ORCH = "#b7791f"; SCHED = "#2f6f4f"; WORK = "#3a5f8a"


def esc(s): return html.escape(str(s))


def box(x, y, w, h, fill, text, sub="", rx=8, tcol=INK, scol=MUTE, ts=15, anchor="middle", stroke=LINE):
    cx = x + w / 2 if anchor == "middle" else x + 12
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>']
    ty = y + (h / 2 + ts / 2 - 3 if not sub else h / 2 - 4)
    out.append(f'<text x="{cx}" y="{ty:.0f}" fill="{tcol}" font-size="{ts}" font-weight="600" text-anchor="{anchor}" font-family="DejaVu Sans, sans-serif">{esc(text)}</text>')
    if sub:
        out.append(f'<text x="{cx}" y="{y + h/2 + 14:.0f}" fill="{scol}" font-size="11.5" text-anchor="{anchor}" font-family="DejaVu Sans Mono, monospace">{esc(sub)}</text>')
    return "".join(out)


def label(x, y, text, col=INK, ts=13, anchor="start", weight="400", mono=False):
    fam = "DejaVu Sans Mono, monospace" if mono else "DejaVu Sans, sans-serif"
    return f'<text x="{x}" y="{y}" fill="{col}" font-size="{ts}" font-weight="{weight}" text-anchor="{anchor}" font-family="{fam}">{esc(text)}</text>'


def arrow(x1, y1, x2, y2, col=MUTE, w=1.6, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}"{d} marker-end="url(#arr)"/>')


# ============================ Panel 1: L0-L6 level model ============================
def panel_level():
    h = 340
    o = [f'<text x="30" y="34" fill="{INK}" font-size="20" font-weight="700" font-family="DejaVu Sans">1 · Level model — the 7-layer hierarchy (L0–L6)</text>']
    rows = [
        ("L6", "CLOS2 / Cluster", "full cluster (N6 super-nodes)", "Worker(level=5) ×N", HOST),
        ("L5", "CLOS1 / SuperNode", "super-node (N5 pods)", "Worker(level=4) ×N", HOST),
        ("L4", "POD / Pod", "pod (4 hosts)", "Worker(level=3) ×N + Sub ×M", HOST),
        ("L3", "HOST / Node", "one host (16 chips + M subs)", "ChipWorker ×N + SubWorker ×M", HOST),
        ("L2", "CHIP / Processor", "one NPU chip (shared GM)", "Host.so + AICPU.so + AICore.o", ON_DEV),
        ("L1", "DIE / L2Cache", "chip die", "hardware-managed", ON_DEV),
        ("L0", "CORE / AIV, AIC", "individual compute core", "hardware-managed", ON_DEV),
    ]
    y0, rh = 56, 36
    for i, (lv, name, unit, comp, accent) in enumerate(rows):
        y = y0 + i * rh
        o.append(f'<rect x="30" y="{y}" width="40" height="{rh-6}" rx="5" fill="{accent}"/>')
        o.append(label(50, y + 21, lv, INK, 14, "middle", "700"))
        o.append(label(86, y + 21, name, INK, 14, weight="600"))
        o.append(label(300, y + 21, unit, MUTE, 12.5))
        o.append(label(600, y + 21, comp, INK, 12.5, mono=True))
    # boundary line between L2 and L3 (after row idx 3)
    by = y0 + 4 * rh - 3
    o.append(f'<line x1="30" y1="{by}" x2="{W-200}" y2="{by}" stroke="{STRONG}" stroke-width="2" stroke-dasharray="7 4"/>')
    o.append(label(W-195, by - 6, "◄ L2 BOUNDARY", STRONG, 12, "start", "700"))
    # world brackets
    o.append(label(W-195, y0 + 24, "HOST / CLUSTER", HOST, 12, "start", "700"))
    o.append(label(W-195, y0 + 40, "Orch+Sched+Worker", MUTE, 11, mono=True))
    o.append(label(W-195, y0 + 54, "IPC · RoCE · HCCS", MUTE, 11, mono=True))
    o.append(label(W-195, by + 40, "ON-DEVICE", ON_DEV, 12, "start", "700"))
    o.append(label(W-195, by + 56, "shared GM + atomics", MUTE, 11, mono=True))
    return "".join(o), h


# ============== Panel 2: three engine components + L2 three-program model ==============
def panel_engine():
    h = 470
    o = [f'<text x="30" y="34" fill="{INK}" font-size="20" font-weight="700" font-family="DejaVu Sans">2 · Engine components (L3+) and the L2 three-program model</text>']
    # three engine boxes
    o.append(box(30, 56, 360, 96, PANEL, "ORCHESTRATOR", "Orch thread · DAG builder", stroke=ORCH))
    o.append(label(46, 118, "Ring · TensorMap · Scope", MUTE, 12, mono=True))
    o.append(label(46, 136, "submit_next_level(c, args, cfg)", MUTE, 12, mono=True))
    o.append(box(410, 56, 360, 96, PANEL, "SCHEDULER", "Scheduler thread · DAG executor", stroke=SCHED))
    o.append(label(426, 118, "wiring → ready → completion queues", MUTE, 12, mono=True))
    o.append(label(426, 136, "moves slot ids; never reads task data", MUTE, 12, mono=True))
    o.append(box(790, 56, 360, 96, PANEL, "WORKER", "Worker threads · execution", stroke=WORK))
    o.append(label(806, 118, "WorkerManager + WorkerThread pool", MUTE, 12, mono=True))
    o.append(label(806, 136, "shm mailbox → forked child → poll", MUTE, 12, mono=True))
    o.append(arrow(390, 104, 410, 104, ORCH, 2))
    o.append(arrow(770, 104, 790, 104, SCHED, 2))
    o.append(label(580, 100, "wiring", MUTE, 10, "middle", mono=True))
    o.append(label(960, 100, "dispatch", MUTE, 10, "middle", mono=True))
    o.append(arrow(790, 132, 770, 132, WORK, 1.4, "4 3"))
    o.append(label(960, 146, "◄ completion (slot, outcome)", MUTE, 10, "middle", mono=True))

    # L2 three-program model below
    o.append(label(30, 196, "At L2 the Worker leaf = ChipWorker driving three on-device programs:", MUTE, 13))
    o.append(box(30, 214, 1120, 40, PANEL, "Python application / SceneTestCase  —  nanobind · ChipWorker (dlopen host.so) · RuntimeBuilder / KernelCompiler", "", ts=13))
    o.append(arrow(590, 254, 590, 280, MUTE, 2))
    o.append(box(30, 286, 540, 64, PANEL, "Host Runtime (C++ .so)", "DeviceRunner · MemoryAllocator · C API", stroke=WEAK))
    o.append(box(610, 286, 540, 64, PANEL, "Binary data", "AICPU .so  +  AICore .o   (dlopen'd / launched)", stroke=LINE))
    o.append(arrow(300, 350, 300, 378, MUTE, 2))
    o.append(arrow(880, 350, 880, 378, MUTE, 2))
    o.append(box(30, 384, 1120, 70, "#10222b", "Ascend device", "", stroke=ON_DEV))
    o.append(label(60, 412, "AICPU: task scheduler loop", INK, 13))
    o.append(label(60, 432, "AICore (AIV/AIC): compute kernels", INK, 13))
    o.append(label(720, 412, "◄── handshake buffers ──►", MUTE, 12, "middle", mono=True))
    o.append(label(720, 432, "aicpu_ready · aicore_done · task ptr", MUTE, 11, "middle", mono=True))
    return "".join(o), h


# ==================== Panel 3: Rust-suitability map ====================
def panel_rust():
    rows = [
        ("L3–L6 Python orchestration", "Python", WEAK, "░ keep Python", "ERGONOMICS"),
        ("Scheduler (queues, dispatch loop)", "C++", STRONG, "██ STRONG", "CONCURRENCY-SAFETY"),
        ("Orchestrator (Ring · TensorMap · Scope)", "C++", STRONG, "██ STRONG", "LIFETIME-SAFETY"),
        ("Remote L3 wire codec / endpoint", "C++", STRONG, "██ STRONG", "PARSING-SAFETY"),
        ("WorkerManager / mailbox dispatch", "C++", MOD, "▓ MODERATE", "CONCURRENCY-SAFETY"),
        ("Host Runtime / DeviceRunner / C API", "C++", WEAK, "░ weak", "FFI-COST"),
        ("nanobind bindings", "C++", WEAK, "░ weak", "INTEROP"),
        ("AICPU scheduler kernel (device)", "C++", WEAK, "░ blocked", "TOOLCHAIN"),
        ("AICore compute kernel (PTO ISA)", "C++/PTO", WEAK, "░ no", "NO-BACKEND"),
    ]
    rh = 34
    h = 70 + len(rows) * rh + 30
    o = [f'<text x="30" y="34" fill="{INK}" font-size="20" font-weight="700" font-family="DejaVu Sans">3 · Rust-suitability map — dominant reason per component</text>']
    o.append(label(30, 56, "COMPONENT", MUTE, 12, "start", "700"))
    o.append(label(560, 56, "TODAY", MUTE, 12, "start", "700"))
    o.append(label(660, 56, "RUST?", MUTE, 12, "start", "700"))
    o.append(label(830, 56, "DOMINANT REASON", MUTE, 12, "start", "700"))
    y0 = 66
    for i, (comp, today, col, verdict, reason) in enumerate(rows):
        y = y0 + i * rh
        if i == 1:  # divider before the host engine block
            o.append(f'<line x1="30" y1="{y-2}" x2="{W-30}" y2="{y-2}" stroke="{LINE}" stroke-width="1"/>')
        if i == 7:  # device boundary
            o.append(f'<line x1="30" y1="{y-2}" x2="{W-30}" y2="{y-2}" stroke="{ON_DEV}" stroke-width="2" stroke-dasharray="7 4"/>')
            o.append(label(W-34, y+13, "L2 device boundary", ON_DEV, 11, "end", "700"))
        o.append(f'<rect x="30" y="{y}" width="8" height="{rh-8}" rx="2" fill="{col}"/>')
        o.append(label(50, y + 18, comp, INK, 13))
        o.append(label(560, y + 18, today, MUTE, 12, mono=True))
        o.append(label(660, y + 18, verdict, col if col != WEAK else MUTE, 12, mono=True, weight="600"))
        o.append(label(830, y + 18, "[" + reason + "]", INK if col == STRONG else MUTE, 12, mono=True))
    o.append(label(30, y0 + len(rows)*rh + 18,
                   "Strong targets = the host coordination core: races, slot-lifetime UAF, untrusted wire bytes — compile-time-eliminated by Rust.",
                   MUTE, 12))
    return "".join(o), h


panels = [panel_level(), panel_engine(), panel_rust()]
gap = 28
total_h = sum(h for _, h in panels) + gap * (len(panels) + 1)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" viewBox="0 0 {W} {total_h}" font-family="DejaVu Sans, sans-serif">',
    f'<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{MUTE}"/></marker></defs>',
    f'<rect width="{W}" height="{total_h}" fill="{BG}"/>',
    f'<text x="30" y="-6" fill="{INK}"></text>',
]
cy = gap
for frag, hh in panels:
    parts.append(f'<rect x="14" y="{cy-10}" width="{W-28}" height="{hh+18}" rx="10" fill="{PANEL}" opacity="0.35"/>')
    parts.append(f'<g transform="translate(0,{cy})">{frag}</g>')
    cy += hh + gap
parts.append("</svg>")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.svg")
open(out, "w").write("\n".join(parts))
print("wrote", out, f"({total_h}px tall)")
