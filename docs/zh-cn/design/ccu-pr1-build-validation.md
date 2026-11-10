# A5 CCU 执行层

CCU 会话支持 A5、CANN 9.3 和两个通信 rank。Host 准备通信与内存，设备可以在同一个会话中连续发起请求。

## 模块与请求

- Host 用 `CcuSession::register_functions` 提交嵌入的 ELF 模块和入口名，列表顺序就是从 0 开始的 `ccu_id`。两个 rank 使用同一份有序定义，发布时会核对模块摘要。
- 模块导出 `simpler_ccu_module_abi_version()`，返回 `CCU_POSITIONAL_ABI_VERSION`。入口接收 `CcuBuildContext *`，按位置读取 tensor 和 scalar，并记录搬运、比较和同步操作。
- Host 先登记设备内存范围，再发布会话。设备用 `ccu_id` 选中函数，将地址、长度和标量填入 `CcuPositionalArguments`。执行层验证地址和长度，从已登记范围取得访问 token，再写入请求槽。
- 完成必须匹配会话代次、请求序号和函数编号。设备读者用完后才可退休请求并复用槽位。Host 关闭时先阻止新请求，并在读者退出且 CCU 服务停止后释放资源。停止未确认时保留资源，调用方可重试 `close()`。
- 模块构建 ABI 保持为 2。设备镜像 ABI 为 3，需用同一份头文件重编 Host 和设备代码。调度器与 Worker 接入分别见 [#2955](https://github.com/hw-native-sys/simpler/pull/2955) 和 [#2956](https://github.com/hw-native-sys/simpler/pull/2956)。

## 构建与 CPU 测试

在装有 CANN 9.3、C++17 编译器、CMake、OpenSSL 开发包和 GoogleTest 的 Linux 上，从仓库根目录执行。若 GoogleTest 安装在独立目录，可追加 `-DCMAKE_PREFIX_PATH=<GTEST_INSTALL>`。

```bash
source <CANN_INSTALL>/set_env.sh
cmake -S src/a5/platform/onboard/host/ccu -B build/ccu \
  -DSIMPLER_CCU_BUILD_TESTS=ON
cmake --build build/ccu --parallel 4
ctest --test-dir build/ccu --output-on-failure
cmake --install build/ccu --prefix <INSTALL_DIR>
```

Host 库与 `ccu_trigger.o` 同目录安装。关闭会话时用该 AIV 产物唤醒服务，让服务执行停止命令。

## 双卡复现

在已分配两张 A5 卡的任务内执行，`TASK_DEVICE` 为逗号分隔的卡号。使用同一套程序跑 16 轮或 1000 轮。16 轮覆盖两遍 8 个偏移，两种轮数都校验目标值、源值、保护区和正常关闭。

```bash
IFS=, read -r device0 device1 <<< "$TASK_DEVICE"
rounds=16  # 专项验收改为 1000
out=$(mktemp -d)
mkdir "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
run_rank() {
    build/ccu/test_ccu_session_onboard "$1" "$2" "$out/rendezvous" "$rounds" \
        build/ccu/ccu_session_onboard.o build/ccu/libccu_positional_module.so \
        >"$out/rank$2.log" 2>&1
}
run_rank "$device0" 0 & pid0=$!
run_rank "$device1" 1 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
cat "$out/rank0.log" "$out/rank1.log"
test "$status" -eq 0
```

在共享机器上，先运行 `.claude/skills/onboard-arch-precheck/check.sh a5`，再用 `task-submit --device auto --device-num 2` 提交上述双卡命令。
