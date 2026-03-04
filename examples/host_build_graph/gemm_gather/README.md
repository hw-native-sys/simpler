# GEMM + Gather Example (a2a3)

本 case 包含两个 kernel：一个计算 kernel（GEMM）、一个通信 kernel（Gather），面向 **a2a3** 平台。任务依赖：先计算（GEMM）再通信（Gather），即 t0 → t1。

## 1. pto-comm-isa 与 simpler-PTO 自带 _deps/pto-isa 的差异说明

### 1.1 结论（a2a3 tgather 相关）

- **a2a3 下 `tests/npu/a2a3/src/st/testcase/tgather` 目录**：  
  **pto-comm-isa** 与 **simpler-PTO 自带 `examples/scripts/_deps/pto-isa`** 中对应路径下的实现 **内容一致**（同一套 tgather_kernel.cpp / tgather_common.h）。
- 即：当前 a2a3 的 TGATHER 用例，两份仓库在实现上无差异，本 case 的 Gather 以 pto-comm-isa 的 a2a3 tgather 为参考即可，头文件与 PTO_ISA_ROOT 使用 simpler-PTO 自带的 _deps/pto-isa 即可。

### 1.2 两套仓库的定位与区别（概览）

| 项目 | 说明 |
|------|------|
| **simpler-PTO 自带 `_deps/pto-isa`** | 来自 `gitcode.com/cann/pto-isa`，由 README 要求克隆到 `examples/scripts/_deps/pto-isa`，作为 **PTO ISA 头文件与接口** 的来源，供 AIC/AIV kernel 编译使用。 |
| **pto-comm-isa** | 独立仓库（PTO Tile Library），除与 pto-isa 同源的 include 外，还包含 **kernels/**、**tests/**（含 a2a3/a5/cpu 等）、**docs/** 等，用于参考实现与测试。 |

- **Include / API**：两边在 a2a3 相关头文件（如 `pto-inst.hpp`、TGATHER 等）上同源或兼容，本 case 仅依赖 _deps/pto-isa 的 include。
- **测试与用例**：pto-comm-isa 的 `tests/npu/a2a3/src/st/testcase/tgather` 作为 **实现参考**（尤其是 runTGather1D、shape、流水）；本 case 的 kernel 写在 simpler-PTO 内，编译时用 _deps/pto-isa。

## 2. Case 说明

- **GEMM**：64×64 float 块乘，复用与 bgemm 同构的实现（AIC），单 task。
- **Gather**：1D 索引 gather，`out = src0[src1]`，shape 与 pto-comm-isa a2a3 tgather 一致（src0: 32×1024 float，src1: 16×64 int32，out: 16×64 float），AIV，单 task。
- **任务依赖**：t0（GEMM）→ t1（Gather），先计算再通信。

## 3. 目录与运行

```
gemm_gather/
├── README.md
├── golden.py
└── kernels/
    ├── kernel_config.py
    ├── orchestration/
    │   └── gemm_gather_orch.cpp
    ├── aic/
    │   └── kernel_gemm.cpp
    └── aiv/
        ├── tgather_common.h
        └── kernel_gather.cpp
```

**a2a3 运行示例**（需 Ascend 设备与 CANN 环境）：

```bash
python examples/scripts/run_example.py \
  -k examples/host_build_graph/gemm_gather/kernels \
  -g examples/host_build_graph/gemm_gather/golden.py \
  -p a2a3 -d <device_id>
```

## 4. Shape 约定（与 comm-isa 一致）

- **GEMM**：A(64,64), B(64,64), C(64,64)，float。
- **Gather**：src0 (32, 1024) float，src1 (16, 64) int32（索引），out (16, 64) float。
