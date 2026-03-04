# 多卡独立跑计算算子 — 修改计划（multi_bgemm）

目标：新增一个 case，**仅跑计算 kernel**（与 bgemm 完全一致），**一次调用**可在 **2 张或 4 张卡**上各自独立跑同一套 graph，**无卡间同步、无通信、无依赖**。

---

## 一、目标与约束

- **Case 名**：`multi_bgemm`（路径：`examples/host_build_graph/multi_bgemm/`）
- **单卡逻辑**：与现有 `bgemm` 完全一致（同一套 orchestration、同一套 kernel：GEMM + tile_add，同一套 golden）
- **多卡方式**：一次 `run_example.py` 调用可指定 2 或 4 张卡；每张卡独立跑同一 case，卡间无同步、无通信
- **不引入**：HCCL、comm_gather、fork 建联等；沿用现有 host_build_graph 单卡 Runtime 流程

---

## 二、实现思路

- 每张卡执行的都是**同一套** host_build_graph 流程：build runtime → 编 orchestration + kernels → `set_device(device_id)` → 对同一组输入跑 `initialize` + `launch_runtime` → `finalize` → 与 golden 比对。
- 多卡实现方式：**多进程**。主进程根据 `n_devices`（2 或 4）起 N 个子进程，每个子进程执行一次「单卡 run」：即当前已有的 `run_example.py -k ... -g ... -d <device_id>` 流程；主进程汇总 N 个子进程退出码，全部为 0 才认为通过。
- 这样无需改 Runtime、不改 bindings、不改单卡逻辑，仅在一层「调度」上做多卡扩展。

---

## 三、端到端需修改/新增内容

### 3.1 新增 case 目录 `multi_bgemm`

| 项 | 说明 |
|----|------|
| **目录** | `examples/host_build_graph/multi_bgemm/` |
| **与 bgemm 关系** | 与 bgemm 同构：同一套编排与 kernel、同一套 golden；仅配置上增加「多卡」相关项。 |
| **建议做法** | 从 bgemm 拷贝以下内容到 multi_bgemm，再按下面调整配置与文档：<br>• `kernels/`（含 `kernel_config.py`、`orchestration/`、`aic/`、`aiv/`）<br>• `golden.py`<br>• `README.md`（重写为 multi_bgemm 说明） |

**需要改动的仅**：

- **`kernels/kernel_config.py`**
  - 在 `RUNTIME_CONFIG` 中增加 `n_devices`，默认 `2`（或 `4`，由你定）；其余（ORCHESTRATION、KERNELS）与 bgemm 保持一致。
- **`README.md`**
  - 说明本 case 为「多卡独立跑 bgemm」，支持 2/4 卡；示例命令用 `--n-devices 2` 或 `--n-devices 4`。

### 3.2 命令行与 code_runner 入参

| 位置 | 修改内容 |
|------|----------|
| **run_example.py** | 新增可选参数 `--n-devices`（类型 `int`，默认 `None`）。解析后传入 `create_code_runner(..., n_devices=args.n_devices)`。 |
| **code_runner.create_code_runner** | 增加形参 `n_devices=None`，并传给 `CodeRunner`。 |
| **CodeRunner.__init__** | 增加 `n_devices` 参数。逻辑：若调用方传入 `n_devices is not None`，则用传入值；否则从 `kernel_config.RUNTIME_CONFIG.get("n_devices", 1)` 读取，默认 `1`。这样既支持「按 case 默认多卡」，也支持「命令行覆盖」。 |

### 3.3 code_runner.run() 中「多卡分支」

- **触发条件**：在 `run()` 开头（在 `comm_gather` 分支之后、正常单卡 build 之前）判断：若 `self.n_devices > 1`，则走「多卡独立跑」分支，直接 return，不再走下面单卡 build/launch。
- **多卡分支逻辑**：
  1. 解析当前脚本所在目录，得到 `run_example.py` 的路径（与 code_runner 同目录：`Path(__file__).resolve().parent / "run_example.py"`）。
  2. 对 `device_id in range(self.n_devices)` 依次（或并行，见下）执行子进程：
     - 命令：`[sys.executable, str(run_example.py), "-k", str(self.kernels_dir), "-g", str(self.golden_path), "-d", str(device_id), "-p", self.platform]`
     - 可选：把当前 `--all` / `--case` / `--log-level` 等与单次 run 相关的参数一并传入，保证与单卡行为一致。
  3. 等待所有子进程结束；若任一子进程 `returncode != 0`，则抛出异常或置失败，并带上设备 id 信息；若全部为 0，则视为通过。
- **并行 vs 串行**：为简单起见先做成**串行**（按 device 0,1,... 顺序跑），避免多进程同时 build 争抢；若你希望 2/4 卡并行跑，可再改为 `concurrent.futures.ProcessPoolExecutor` 或 `multiprocessing.Pool`，但需注意编译/资源竞争，建议首版串行。
- **Build 次数**：每个子进程都会完整跑一遍 run_example（含 build runtime、编 orchestration、编 kernel），因此会 build N 次；实现简单，首版接受该冗余，后续若有需要可再做「只跑不编」的优化。

### 3.4 不需要改动的部分

- **Runtime / bindings / 单卡 launch**：不变。
- **单卡 case（如 bgemm）**：不加 `n_devices` 时行为与现在完全一致（`n_devices` 默认 1，不进入多卡分支）。
- **comm_gather**：不受影响，仍走 `_run_comm_gather()`。

---

## 四、文件与配置清单（实施顺序）

| 步骤 | 操作 |
|------|------|
| 1 | 新建目录 `examples/host_build_graph/multi_bgemm/`。 |
| 2 | 从 `bgemm` 拷贝 `golden.py`、`kernels/`（整个目录：kernel_config.py、orchestration/、aic/、aiv/）到 `multi_bgemm/`。 |
| 3 | 修改 `multi_bgemm/kernels/kernel_config.py`：增加 `RUNTIME_CONFIG = { "runtime": "host_build_graph", "n_devices": 2 }`（或 4）；若 bgemm 当前无 `RUNTIME_CONFIG`，则仅在 multi_bgemm 中显式写出，保持与 bgemm 相同的 ORCHESTRATION、KERNELS。 |
| 4 | 编写 `multi_bgemm/README.md`：说明本 case 为多卡独立跑计算（与 bgemm 同逻辑），示例命令 `--n-devices 2` / `--n-devices 4`，无卡间同步与通信。 |
| 5 | **run_example.py**：增加 `--n-devices` 参数，并传入 `create_code_runner(..., n_devices=args.n_devices)`。 |
| 6 | **code_runner.py**：`create_code_runner` 增加 `n_devices` 参数；`CodeRunner.__init__` 中读取并保存 `self.n_devices`（来自参数或 `RUNTIME_CONFIG`，默认 1）。 |
| 7 | **code_runner.run()**：在 `comm_gather` 分支之后、单卡 build 之前，若 `self.n_devices > 1`，则执行「多卡子进程」逻辑（循环或并行启动 N 个 `run_example.py -d 0..N-1`），全部成功则 return，否则抛错。 |

---

## 五、运行示例（计划通过后）

```bash
# 2 张卡独立跑 multi_bgemm（每卡跑同一套 bgemm 计算）
python examples/scripts/run_example.py \
  -k examples/host_build_graph/multi_bgemm/kernels \
  -g examples/host_build_graph/multi_bgemm/golden.py \
  --n-devices 2

# 4 张卡（若 kernel_config 默认 n_devices=4 也可不写）
python examples/scripts/run_example.py \
  -k examples/host_build_graph/multi_bgemm/kernels \
  -g examples/host_build_graph/multi_bgemm/golden.py \
  --n-devices 4
```

每张卡使用与 bgemm 相同的输入（各自 `generate_inputs` 一份，或子进程内各自生成，因随机数可能不同；若需完全一致可后续改为主进程生成写文件再传入，首版可保持各进程独立 `generate_inputs`，golden 仍按同一规则校验）。

---

## 六、小结

- **新 case**：`multi_bgemm`，与 bgemm 同编排、同 kernel、同 golden，仅配置多卡数。
- **入口**：`run_example.py` 增加 `--n-devices`；code_runner 支持 `n_devices` 从配置或命令行传入。
- **执行**：`n_devices > 1` 时用子进程对每张卡跑一次「单卡 run_example」流程，无卡间同步与通信，全部成功即通过。

确认该计划 OK 后，再按上述步骤改代码。
