# multi_bgemm — 多卡独立跑 BGEMM（无通信）

在 2 张或 4 张卡上**并行**跑与 **bgemm** 完全相同的计算（C = A @ B，4x4x4 grid、64x64 tile），每张卡独立执行，无卡间同步、无通信。

## 用法

```bash
# 2 张卡，起始卡号 0 → device 0, 1 并行
python examples/scripts/run_example.py \
  -k examples/host_build_graph/multi_bgemm/kernels \
  -g examples/host_build_graph/multi_bgemm/golden.py \
  --n-devices 2 --first-device 0

# 4 张卡，起始卡号 4 → device 4, 5, 6, 7 并行
python examples/scripts/run_example.py \
  -k examples/host_build_graph/multi_bgemm/kernels \
  -g examples/host_build_graph/multi_bgemm/golden.py \
  --n-devices 4 --first-device 4
```

- **--n-devices**：卡数（不传时使用 kernel_config 中 `RUNTIME_CONFIG["n_devices"]`，默认 2）。
- **--first-device**：起始卡号（不传时使用 `RUNTIME_CONFIG["first_device_id"]`，默认 0）。
- 设备号区间：`[first-device, first-device + n-devices)`。

单卡时可不传 `--n-devices` 或传 `--n-devices 1`，则走普通单卡流程。

## 行为说明

- 与 **bgemm** 使用同一套 orchestration（`build_bgemm_graph`）、同一套 kernel（GEMM + tile_add）、同一套 golden。
- **编译与运行分离**：主进程先 `compile()` 一次，创建 N 个 CodeRunner（传入 compiled_artifacts），通过 `ProcessPoolExecutor` 多进程并行执行各 `runner.run()`，**无重复编译**。
- 不引入 HCCL、通信算子或建联逻辑；与后续多卡通信方案兼容（通信 case 将使用独立 C++ 入口）。

## 目录结构

```
multi_bgemm/
├── golden.py
├── README.md
└── kernels/
    ├── kernel_config.py       # 含 RUNTIME_CONFIG (n_devices, first_device_id)
    ├── orchestration/
    │   └── bgemm_orch.cpp
    ├── aic/
    │   └── kernel_gemm_tile.cpp
    └── aiv/
        └── kernel_tile_add.cpp
```
