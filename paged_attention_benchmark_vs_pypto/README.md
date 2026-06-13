# Paged Attention Benchmark

```bash
python paged_attention_benchmark_vs_pypto/benchmark_spmd_paged_attention_compare.py \
  -p a2a3 -d 6 --rounds 10 --warmup 2 --validate-results
```

```bash
python paged_attention_benchmark_vs_pypto/benchmark_spmd_paged_attention_compare.py \
  -p a2a3 -d 6 --rounds 10 --warmup 2 --validate-results --shape-suite
```

```bash
python paged_attention_benchmark_vs_pypto/benchmark_spmd_paged_attention_compare.py \
  -p a2a3 -d 6 --rounds 10 --warmup 2 --validate-results \
  --shape b1_h32_kv8_s4096_bs128_fp16
```

```bash
python paged_attention_benchmark_vs_pypto/benchmark_spmd_paged_attention_compare.py \
  -p a2a3 -d 6 --rounds 10 --warmup 2 --skip-highperf --validate-pypto \
  --shape b1_h32_kv8_s8192_bs128_fp16
```

```bash
python paged_attention_benchmark_vs_pypto/benchmark_spmd_paged_attention_compare.py \
  -p a2a3 -d 6 --rounds 10 --warmup 2 --skip-highperf --validate-pypto \
  --shape b1_h32_kv8_s16384_bs128_fp16
```
