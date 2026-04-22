## Usage

```bash
bash ./compile.sh
python ./test_pa_accuracy.py
python ./bench_pa_performance.py
```

## Reference performance

Bandwidth utilization for GQA decoding shapes:

| Case | TFLOP/s | GiB/s |
|------|---------|--------|
| Qwen3-0.6B b1 h16/kv8 kv2048 | 0.24 | 110 |
| Qwen3-8B b1 h32/kv8 kv4096 | 1.02 | 237 |
| Qwen3-8B b1 h32/kv8 kv8192 | 1.65 | 384 |
| Qwen3-8B b64 h32/kv8 kv2048 | 4.48 | 1045 |

Compare to library performance:

| Case | Standalone ms | ATB v2 ms | Speedup (ATB / standalone) |
|------|---------------|-----------|------------------------------|
| Qwen3-0.6B b1 h16/kv8 kv2048 | 0.0709 | 0.1156 | 1.63× |
| Qwen3-1.7B b1 h16/kv8 kv4096 | 0.0676 | 0.1000 | 1.48× |
| Qwen3-4B b1 h32/kv8 kv2048 | 0.0683 | 0.0980 | 1.43× |
| Qwen3-8B b1 h32/kv8 kv4096 | 0.0659 | 0.0881 | 1.34× |
| Qwen3-8B b1 h32/kv8 kv8192 | 0.0814 | 0.0868 | 1.07× |
| Qwen3-14B b1 h40/kv8 kv2048 | 0.0662 | 0.0860 | 1.30× |
| Qwen3-32B b1 h64/kv8 kv2048 | 0.0666 | 0.0827 | 1.24× |
| MHA b1 h32/kv32 kv2048 | 0.0671 | 0.0895 | 1.33× |
| Qwen3-8B b4 h32/kv8 kv2048 | 0.0707 | 0.0965 | 1.36× |
| Qwen3-8B b8 h32/kv8 kv2048 | 0.0877 | 0.1120 | 1.28× |
| Qwen3-8B b16 h32/kv8 kv2048 | 0.1155 | 0.1448 | 1.25× |
| Qwen3-8B b32 h32/kv8 kv2048 | 0.2635 | 0.2459 | 0.93× |
| Qwen3-8B b64 h32/kv8 kv2048 | 0.4793 | 0.4749 | 0.99× |