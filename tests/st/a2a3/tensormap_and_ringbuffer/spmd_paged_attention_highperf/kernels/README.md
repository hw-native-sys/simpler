## Usage

```bash
bash ./compile.sh
python ./test_pa_accuracy.py
python ./bench_pa_performance.py
```

`bench_pa_performance.py` compares the standalone `pa_lib.so` kernel against
`torch_npu.npu_incre_flash_attention` with **paged KV** (`block_table`, `actual_seq_lengths`,
`block_size`), using the same calling pattern as `ifa/bench_ifa_gpa_paged.py`.

## Reference performance

Bandwidth utilization for GQA decoding shapes (standalone `pa_lib.so`, fp16, measured on one
910B-class device; numbers vary by card and load):

| Case | TFLOP/s | GiB/s |
|------|---------|--------|
| Qwen3-0.6B b1 h16/kv8 kv2048 | 0.22 | 105 |
| Qwen3-8B b1 h32/kv8 kv4096 | 0.97 | 226 |
| Qwen3-8B b1 h32/kv8 kv8192 | 1.80 | 420 |
| Qwen3-8B b64 h32/kv8 kv2048 | 4.50 | 1050 |

Compare to `torch_npu.npu_incre_flash_attention` (paged KV, same shapes):

| Case | Standalone ms | IFA ms | Speedup (IFA / standalone) |
|------|---------------|--------|------------------------------|
| Qwen3-0.6B b1 h16/kv8 kv2048 | 0.0747 | 0.0767 | 1.03× |
| Qwen3-1.7B b1 h16/kv8 kv4096 | 0.0687 | 0.0782 | 1.14× |
| Qwen3-4B b1 h32/kv8 kv2048 | 0.0736 | 0.0789 | 1.07× |
| Qwen3-8B b1 h32/kv8 kv4096 | 0.0693 | 0.0781 | 1.13× |
| Qwen3-8B b1 h32/kv8 kv8192 | 0.0745 | 0.0767 | 1.03× |
| Qwen3-14B b1 h40/kv8 kv2048 | 0.0698 | 0.0792 | 1.14× |
| Qwen3-32B b1 h64/kv8 kv2048 | 0.0689 | 0.0767 | 1.11× |
| MHA b1 h32/kv32 kv2048 | 0.0705 | 0.0752 | 1.07× |
| Qwen3-8B b4 h32/kv8 kv2048 | 0.0732 | 0.0784 | 1.07× |
| Qwen3-8B b8 h32/kv8 kv2048 | 0.0921 | 0.0818 | 0.89× |
| Qwen3-8B b16 h32/kv8 kv2048 | 0.1219 | 0.1024 | 0.84× |
| Qwen3-8B b32 h32/kv8 kv2048 | 0.2637 | 0.2725 | 1.03× |
| Qwen3-8B b64 h32/kv8 kv2048 | 0.4770 | 0.5337 | 1.12× |

Table source: `python bench_pa_performance.py --warmup 5 --iters 20`
