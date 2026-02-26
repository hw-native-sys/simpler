Scaffold a new example at $ARGUMENTS.

1. Read `.ai-instructions/coding/testing.md` and `examples/CLAUDE.md` for the expected structure
2. Parse $ARGUMENTS as `<runtime_variant>/<example_name>` (e.g., `tensormap_and_ringbuffer/my_kernel`)
3. Create the directory structure:
   ```
   examples/<runtime_variant>/<example_name>/
     golden.py
     kernels/
       kernel_config.py
       aiv/
       orchestration/
   ```
4. Populate `golden.py` with stub `generate_inputs(params)` and `compute_golden(tensors, params)` functions
5. Populate `kernel_config.py` with template `KERNELS`, `ORCHESTRATION`, and `RUNTIME_CONFIG` matching the chosen runtime
6. Add a stub orchestration `.cpp` file under `kernels/orchestration/`
7. Report what was created and how to run it:
   `python examples/scripts/run_example.py -k examples/<path>/kernels -g examples/<path>/golden.py -p a2a3sim`
