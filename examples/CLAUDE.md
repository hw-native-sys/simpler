# Examples

Each example demonstrates an end-to-end kernel running on a specific runtime variant. Examples are auto-discovered by `ci.sh` — no registration needed.

## Directory Layout

```
<runtime_variant>/<example_name>/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS, ORCHESTRATION, RUNTIME_CONFIG
    aic/                 # AICore-CUBE kernel sources (optional)
    aiv/                 # AICore-VECTOR kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

## Adding a New Example

1. Create `examples/<runtime>/<name>/` with the layout above
2. Implement `generate_inputs(params)` and `compute_golden(tensors, params)` in `golden.py`
3. Define `KERNELS`, `ORCHESTRATION`, and `RUNTIME_CONFIG` in `kernel_config.py`
4. Test with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p a2a3sim`

See `.ai-instructions/coding/testing.md` for the full golden test pattern reference.
