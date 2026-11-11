# Qwen3-14B 40-layer decode on percounter

The same scene as the
[`host_build_graph` case](../../host_build_graph/qwen3_14b_decode/README.md),
scheduled by `percounter` instead. `main.py` delegates to the TMR standalone
driver and changes only the runtime name, reusing that case's orchestration
source unchanged along with the 37 incore sources, streaming fixture, and
golden implementation. Running one graph through both schedulers is how they
are compared, so the two cases are kept identical in everything else.

Run on A5 hardware through the shared-device queue:

```bash
task-submit --device auto --device-num 1 --run \
  ".claude/skills/onboard-arch-precheck/check.sh a5 && \
   .venv/bin/python examples/a5/percounter/qwen3_14b_decode/main.py \
   -p a5 -d \"\$TASK_DEVICE\" --rounds 100 --skip-golden"
```

`percounter` declares `BUILD_BY_DEFAULT = False`, so a build that names no
runtime does not produce it. Build it first:

```bash
python -m simpler_setup.build_runtimes --platforms a5 --runtime percounter
```

There is no pytest wrapper beside this `main.py`. The scene tests and unit
tests for this runtime land with its CI in a separate change; until then the
case is driven by hand, and nothing here is collected by a scene-test run.
