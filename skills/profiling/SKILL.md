---
name: profiling
description: On-demand guide for profiling PTO Runtime examples on hardware, interpreting swimlane output, and visualizing task execution timelines with Perfetto and Mermaid.
---

# Profiling Skill

## When to Use

Load this skill when:
- Running examples with `--enable-profiling`
- Interpreting `perf_swimlane_*.json` output files
- Generating Chrome Trace Event or Mermaid visualizations
- Analyzing per-function timing breakdowns

## Running with Profiling

```bash
python examples/scripts/run_example.py \
    -k examples/<runtime>/<name>/kernels \
    -g examples/<runtime>/<name>/golden.py \
    -p a2a3 \
    --enable-profiling
```

Output files are written to `outputs/` in the example directory.

## Visualization Tools

### Swimlane Converter (Chrome Trace → Perfetto)

```bash
python tools/swimlane_converter.py outputs/perf_swimlane_*.json -o trace.json
```

Open `trace.json` in [Perfetto UI](https://ui.perfetto.dev/) for an interactive timeline view.

The converter produces:
- Task timelines per core (AIC, AIV, AICPU)
- Task statistics: count, total time, avg, min, max per function
- Color-coded by core type

### Mermaid Flowchart

```bash
python tools/perf_to_mermaid.py outputs/perf_swimlane_*.json
```

Generates a Mermaid flowchart showing task dependencies, styled by core type. Embeddable in GitHub/GitLab markdown.

## Reading the Output

### Console Statistics

After profiling, the console shows per-function timing:

```
Function           Count   Total(us)   Avg(us)   Min(us)   Max(us)
kernel_add         3       45.2        15.1      14.8      15.5
build_graph        1       12.3        12.3      12.3      12.3
```

### Key Metrics

- **Total time** — cumulative execution across all invocations
- **Avg time** — mean per-invocation cost
- **Min/Max spread** — indicates consistency (large spread = investigate)

## Optimization Tips

1. Look for functions with high total time — candidates for optimization
2. Check if AIC and AIV tasks overlap (good) or serialize (bad)
3. Verify AICPU scheduling overhead is small relative to compute time
4. Compare simulation vs hardware times to validate simulator fidelity
