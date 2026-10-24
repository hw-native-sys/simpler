# Host 通信域：单机与双机 Direct MPI 验收

本流程验证 A3 HCCS 上每台机器的一块 CPU DRAM、两块 NPU HBM。
数据面只使用 CANN ACL 和 NPU kernel，不安装或链接 GitCode HIXL/ADXL。

2026-10-23 用户反馈已在真机执行 utpy、单机双卡及双机四卡测试，全部 PASS。
该结果按用户实测报告记录；本仓库尚未附上此次原始日志、精确测试提交和逐项执行数量，
不将下文预期计数冒记为已核对的日志结果，也不据此推断 HCCS 逐跳物理路由。

2026-10-23 已在 A3 单机两卡上用同一 `run_matrix` 核心完成本地 L4＋fork L3 对照：
18 项矩阵（CPU 本地 2、ACL 4、kernel 12）、7 步同址交接、完整窗口与 guard、显式释放全部通过，
`released=true` 且任务退出码为 0。该环境没有 MPI 启动器，因而这组本地证据仅证明单机本体流程，
与上述用户另行完成的单机／双机测试分开记录。
统一 export 表和 identity-only copy 请求后，已用同一矩阵复测通过。

## 验证内容

入口：`examples/workers/l4/host_domain_mpi_direct/main.py`。
一个 MPI 作业有一个 L4 controller，以及一或两个 L3 executor；每个 executor 管理两张 NPU。
CPU 直接访问的只有本机 owner DRAM；访问其他池时指定本机 chip，通过 ACL SDMA 同步搬运。
每张 NPU 的 AIV kernel 使用 TLOAD/TSTORE 主动访问全部池。
Host 池通过普通 Tensor 参数传入，peer HBM 地址从现有 CommContext 的 windowsIn 推导。
不把 SDMA 成功记成 kernel 成功，不尝试跨机 Host-target import。

| 规模 | 域内 allocation | CPU 直接 READ/WRITE | CPU ACL READ/WRITE | kernel READ/WRITE | 矩阵合计 |
| ---- | --------------- | ------------------- | ------------------ | ----------------- | -------- |
| 单机 | 1 DRAM＋2 HBM | 2 | 4 | 12 | 18 |
| 双机 | 2 DRAM＋4 HBM | 4 | 20 | 48 | 72 |

域内每块 HBM 逻辑大小 4096 B，每块 DRAM 逻辑大小 16384 B；对齐 padding 不计入可访问窗口。
Host DRAM 的待测范围为 `[256,1280)`，写入源为 `[1536,2560)`，独立读结果为 `[2816,3840)`，
HBM 全窗快照暂存为 `[8192,12288)`。这些范围不重叠。
Host 与 HBM 的待测范围大小相同，待测范围之外全部内容也参与逐字节验证，包含两侧 guard。
仅统计域 payload 的三个或六个物理 allocation；运行时内部 workspace 不计入此数字。

每个矩阵格子依次执行三个完整 L4 run：重置全部池、执行一个 READ/WRITE、各 owner 校验。
初始化花纹在各 L3 本地生成，CPU 直接写本机 DRAM，再用本机 ACL copy 初始化 HBM。
控制通道只携带 Tensor 描述符、操作编号、范围与 kernel digest；不传输被测 payload 或实际读回字节。
READ 写入独立结果区后比较；WRITE 比较目标完整窗口。
每次校验先检查 Host 全窗，再将本地 HBM 全窗经 ACL 读到 Host 快照区比较，并恢复快照区后再次检查 Host。
校验中的 ACL 回读不替代 kernel 被测操作。

矩阵之后另外执行同址交接，不重新建域、不在交接步骤间重置：
owner CPU 写入 → 本地 NPU0 读后写入 → 本地 NPU1 读后写入 → owner CPU 读回。
双机还加入一张远端 NPU 对同一区间的读写，最后仍由 owner CPU 验证。
所有步骤均有完整窗口校验；交接步骤数单独报告，不混入 18/72 的矩阵计数。

## 两台机器的前置条件

- 两台机器使用同一提交、同一 CANN/驱动支持组合，A3 Fabric/HCCS 跨机链路可用。
- 每台提供两张独占、空闲的 NPU；NUMA 0 支持 Host NUMA/P2P_HUGE 分配。
- 两边安装同一种 MPI 实现，`mpi4py` 与该实现匹配，A 能通过非交互 SSH 启动 B 的 MPI rank。
- 两边安装该 checkout 的 simpler，包括 native 模块和 `tensormap_and_ringbuffer` runtime。
- A 配置 kernel 编译所需的 pto-isa。不要在检查中提前调用 `MPI.Init` 或导入会自动初始化 MPI 的程序。
- 两边相同绝对路径 `/tmp/simpler-host-domain-python` 必须可执行；其内部 checkout 路径可以不同。
- 下面的 `192.0.2.10`、`192.0.2.20` 是占位地址，替换成 A、B 实际可达的数字 IP。

共享服务器必须先按当地调度规范预约两台机器的相关设备。下列硬件命令只适用于已获分配的环境；
不能在要求单卡 task-submit 的服务器上直接占用两卡，或借 MPI 绕过设备调度。
以下 A/B 命令分别在各自 checkout 的隔离 `.venv` 中安装和构建，再检查导入路径；
runtime 构建会校验并记录 pinned PTO-ISA 与 onboard 元数据，仅有 sim runtime 不能运行硬件验收。

### B 机器：准备环境与远端解释器

在 B 上执行，先替换 `SIMPLER_ROOT`：

```bash
export SIMPLER_ROOT=/path/to/B/simpler
cd "$SIMPLER_ROOT"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1
test -d .venv || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export CCACHE_DIR="$PWD/.ccache"
export CMAKE_BUILD_PARALLEL_LEVEL=4
ccache -s
python -m pip install scikit-build-core 'nanobind>=2,<3' cmake ninja
python -m pip install -v --no-build-isolation -e '.[mpi]'
python -c 'from simpler_setup.runtime_builder import RuntimeBuilder; print(RuntimeBuilder("a2a3").get_binaries("tensormap_and_ringbuffer", build=True))'
python -c 'import simpler, _task_interface, mpi4py; print(simpler.__file__); print(_task_interface.__file__); print(mpi4py.__version__)'
command -v mpirun
command -v mpiexec
git rev-parse HEAD
cat > /tmp/simpler-host-domain-python <<EOF_LAUNCHER
#!/usr/bin/env bash
set -e
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1
cd "$SIMPLER_ROOT"
exec "$SIMPLER_ROOT/.venv/bin/python" "\$@"
EOF_LAUNCHER
chmod +x /tmp/simpler-host-domain-python
```

B 不另启动 Python runner、daemon 或第二个 mpirun；等待 A 的一个作业拉起 B 的 executor。

### A 机器：准备环境与同名解释器

在 A 上执行，先替换 `SIMPLER_ROOT`：

```bash
export SIMPLER_ROOT=/path/to/A/simpler
cd "$SIMPLER_ROOT"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1
test -d .venv || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export CCACHE_DIR="$PWD/.ccache"
export CMAKE_BUILD_PARALLEL_LEVEL=4
ccache -s
python -m pip install scikit-build-core 'nanobind>=2,<3' cmake ninja
python -m pip install -v --no-build-isolation -e '.[mpi]'
python -c 'from simpler_setup.runtime_builder import RuntimeBuilder; print(RuntimeBuilder("a2a3").get_binaries("tensormap_and_ringbuffer", build=True))'
python -c 'import simpler, _task_interface, mpi4py; print(simpler.__file__); print(_task_interface.__file__); print(mpi4py.__version__)'
command -v mpirun
command -v mpiexec
git rev-parse HEAD
cat > /tmp/simpler-host-domain-python <<EOF_LAUNCHER
#!/usr/bin/env bash
set -e
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1
cd "$SIMPLER_ROOT"
exec "$SIMPLER_ROOT/.venv/bin/python" "\$@"
EOF_LAUNCHER
chmod +x /tmp/simpler-host-domain-python
ssh -o BatchMode=yes -o ConnectTimeout=10 192.0.2.20 \
  '/tmp/simpler-host-domain-python -c "import simpler, _task_interface; print(simpler.__file__); print(_task_interface.__file__)"'
```

确认两个 `git rev-parse HEAD` 相同，导入路径指向各自的预期 checkout/安装。
不要临时换用其他 checkout 的 native 二进制。

## A 机器：先跑单机两卡对照

只启动 controller 和 A executor，共两个 MPI rank。下列采用 MPICH：

```bash
cd "$SIMPLER_ROOT"
set -o pipefail
python -m examples.workers.l4.host_domain_mpi_direct.main \
  --local-host 192.0.2.10 \
  --python /tmp/simpler-host-domain-python \
  --local-devices 0,1 \
  --mpirun-path "$(command -v mpiexec)" \
  --launcher-family mpich 2>&1 | tee host-domain-single.log
```

使用 Open MPI 时将最后两项替换为 `--mpirun-path "$(command -v mpirun)" --launcher-family openmpi`。
不得在外面再套一层 mpirun。先确认 18 个 `CASE_PASS`，摘要为 `nodes=1`、`domain_pools=3`、`checks=18`，
`cpu_local=2`、`cpu_acl=4`、`kernel=12`、`released=true`，并且整个命令退出码为 0。

## A 机器：双机四卡验收

只在 A 启动一次，共三个 MPI rank；设备编号分别是 A、B 的本地逻辑编号：

```bash
cd "$SIMPLER_ROOT"
set -o pipefail
python -m examples.workers.l4.host_domain_mpi_direct.main \
  --local-host 192.0.2.10 --remote-host 192.0.2.20 \
  --python /tmp/simpler-host-domain-python \
  --local-devices 0,1 --remote-devices 0,1 \
  --mpirun-path "$(command -v mpiexec)" \
  --launcher-family mpich 2>&1 | tee host-domain-two-host.log
```

完整成功需要 72 个 `CASE_PASS`、同址交接全部完成、显式域释放成功，并有摘要：

```json
{"result": "MATRIX_PASS", "nodes": 2, "domain_pools": 6, "checks": 72, "cpu_local": 4, "cpu_acl": 20, "kernel": 48, "handoff_steps": 18, "released": true}
```

最终还必须是 runner/supervisor **退出码 0**，表示其后 Direct MPI executor 关闭也成功。
只出现 `MATRIX_PASS` 后作业异常退出，不能记为完整 PASS。
任何 import/copy/kernel/字节比较/guard/释放失败均应保留完整日志并记录失败；
没有 MPI、只运行 sim、缺卡、超时或 Feature Not Support 均不等同于双机通过。
本流程不采集 HCCS/PCIe 逐端口流量，结果不能证明逐跳物理路由。

## CANN-only 与复用本体验证

样例没有 HIXL/ADXL 头文件、链接指令或引擎初始化；编译调用复用 simpler 的 CANN ACL 后端。
如部署环境存在其他库，应另查实际构建链接记录与动态依赖，不能仅凭系统安装过 HIXL 就断言测试使用它。

`run_matrix(worker, nodes, kernel_handle, route_handle)` 可复用到已初始化的本地 L4＋fork L3 对照。
本地包装调用 `node_orch(..., chip_handle=预先注册的handle)`，远端调用由 digest 解析已分发的 handle。
该入口执行同一组窗口操作和校验，但本地 fork 结果不能代替 Direct MPI 启动或双机验收。
