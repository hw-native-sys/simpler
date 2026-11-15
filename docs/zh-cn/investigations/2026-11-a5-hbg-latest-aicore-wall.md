# HBG 固定版本的 L0 AICore wall

本表保存 fwic-aicore session 对固定提交 `2481bd9431437b665b2ea81c786eef800ea33ab8` 的测量，
不是 rebase 到后续 main 后的重新测量。对应提交说明：
`Update: service newly ready cohorts before future preparation`。
核对 903 个构建相关文件与目标提交一致，并校验二进制散列后，直接复用已编译运行库。
业务 kernel、输入和 16 个配置沿用前一轮，不重新编译。

## 测试方式

- 只呈现 L0 的实际 kernel 首尾窗口：每卡取第一个实际 kernel begin 到最后一个 kernel end。
  包含两者之间的调度、等待和通信空洞，不包含首尾之外的启动、退出及日志发布。
- 每组 5 次预热、10 次正式测量，保留原有 prefix、输入复位和 numerical golden 校验。
  新版原生 MoE 保留 Host `get_tensor_data` 路径及 0/6、6/0、72/72 三组正确性检查。
- 双卡在被测运行前执行 fwic-graph 的 Device epoch rendezvous，验证双方同步完成。
  每轮分别计算两卡的 kernel 窗口，取较大值后再做十轮平均，不跨卡相减计数器。
- 通过 `task-submit --device auto --device-num N` 自动分配，最大并发 8 张卡。
  下表记录真实设备分配；测试结束后全部释放。
- 本轮 16 组全部通过 golden、时间戳及双卡同步检查，没有使用 L1/L3 数据替代 L0。

## 平均值

单位：ms。

| Case | Implementation | AICore wall (ms) | Device |
| ---- | -------------- | ---------------: | ------ |
| swa | 新版 PyPTO-lib | 0.321653 | 3 |
| c2a_full | 初版 PyPTO-lib | 1.187823 | 7 |
| c2a_full | DSL 半同步 | 1.570328 | 2 |
| c2a_full | DSL 全同步≤64 | 1.602462 | 6 |
| c2a_full | 新版 PyPTO-lib | 0.408528 | 4 |
| c1a_reindex | 新版 PyPTO-lib | 0.401922 | 5 |
| c1a_reuse | 新版 PyPTO-lib | 0.306594 | 6 |
| moe | 初版 PyPTO-lib | 3.442943 | 2,3 |
| moe | DSL 半同步 | 1.523743 | 2,3 |
| moe | DSL 全同步≤64 | 1.559213 | 4,5 |
| moe | 新版 PyPTO-lib | 0.834525 | 0,1 |
| engram | 新版 PyPTO-lib | 1.008130 | 7 |
| layers_19_21 | 初版 PyPTO-lib | 14.332172 | 2,3 |
| layers_19_21 | DSL 半同步 | 9.263651 | 4,5 |
| layers_19_21 | DSL 全同步≤64 | 9.534265 | 6,7 |
| layers_19_21 | 新版 PyPTO-lib | 3.769646 | 4,5 |

## 本地证据

以下路径位于该 session 的独立工作区。原始测量、首尾计数器及同步证据：`artifacts/dfx-budget/graph-latest-l0/`。
版本、源码一致性及二进制复用记录：`graph-latest-source-audit.json`、`graph-latest/build.json`。
同步包装及原始可执行段校验：`graph-latest-gate/manifest.json`。
汇总：`GRAPH_LATEST_RESULTS.json`。Device wall 仅保留为原始诊断数据，不作为本表指标。
