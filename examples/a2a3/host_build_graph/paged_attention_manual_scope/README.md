# paged_attention_manual_scope — host_build_graph

Same computation as `../paged_attention/`, with the automatic same-scope
dependency wiring replaced by explicit task-to-task edges inside
`PTO2_SCOPE(PTO2ScopeMode::MANUAL)`. Read it against the baseline to see what
the automatic mode was deriving.

## The two dependency APIs it demonstrates

| API | Shape | Suited to |
| --- | ----- | --------- |
| `Arg::set_dependencies(buf, n)` | caller owns the buffer, `Arg` stores `(ptr, count)` | codegen, fixed dep sets |
| `L0TaskArgsWithDeps<>::add_dep(id)` | wrapper owns a stack-sized buffer, incremental | hand-written orch, deps assembled across branches |

`SF` and `PV` use the primitive form; `UP` uses the convenience form because its
dep set is conditional — it always takes the `PV` edge, adds the previous
`UP` task when there is one, and adds the alloc task on the last block so the
scratch buffers outlive their final consumer.

Both APIs exist unchanged on `host_build_graph`: the orchestration header is
byte-identical between the two runtimes, so this variant is the same C++ as its
`tensormap_and_ringbuffer` sibling.

## Ring sizing

`host_build_graph` submits the whole graph before the device schedules, so every
task of a case is live at once and the ring must hold all of them — `Case1`
65 792 and `Case2` 32 832, both above the default 16384 window. Each case
carries its own `runtime_env` sizing in `CASES[*]["config"]`.

This is a stronger constraint than the `tensormap_and_ringbuffer` variant faces,
where slots retire as orchestration proceeds and only the *in-flight* depth of a
single MANUAL scope has to fit.

## Cases

`Case1`, `Case2`, `CaseSmall1`, `CaseSmall2`, `CaseVarSeq2`, `CaseVarSeq4`. All
but `CaseSmall1` are `manual`. The upstream `Case3` (`head_dim: 256`) is absent
because it does not produce correct results on either runtime — see
`KNOWN_ISSUES.md`.

## Run

```bash
python examples/a2a3/host_build_graph/paged_attention_manual_scope/test_paged_attention_manual_scope.py \
    -p a2a3 -d 0                                   # CaseSmall1, golden checked
python examples/a2a3/host_build_graph/paged_attention_manual_scope/test_paged_attention_manual_scope.py \
    -p a2a3 -d 0 --manual include --case Case1 --rounds 2
```
