# CSA Mix提前退休：局部确认更快，连续组周转变慢

日期：2026-10-28。结论：撤回，不保留生产改动或绑定该策略的测试。

## 问题与候选

8B close-seal、v5协议的CSA基线L4中，26个Mix任务最后贡献组的全三lane release
到C确认退休，中位间隔为6.267us。fill在确认当前已RELEASED的组之前，先发现新候选。
这会推迟TaskDone及依赖后继，值得针对全核计算空窗验证。

候选在已有pending发布之后、discovery之前，先复用frontier和ordinary槽检查，
再调用原退休与TASK_LAST完成路径；随后照常discovery。提前路径不close、claim或发布下一组。
同轮后段用局部布尔避免重复退休；三lane强ACK、epoch/error、唯一完成和两份candidate credit均保留。
无ABI或原生协议变化，不修改54个业务kernel、2308个执行身份或七项golden。

## 验证和结果

新增24项参数化测试，在旧顺序上16项按预期失败、8项保护通过。候选全部通过，
六个关联C++目标通过。四项旧noREADY时序断言曾调整为READY必须正常准入且身份正确，
原exactly-once验证未删除。严格原样CSA L4通过，保留133条依赖和46733个协议事件。
未将此范围冒充完整设备故障矩阵或全部HBG场景验收。

同卡按A/B/A串行采样，每批3轮×7项golden、5预热+100正式样本：

| 批次 | 完整Device中位数（us） |
| ---- | ---------------------- |
| `idle-prefix8-before-retire-a`：基线 | 2250.4470 |
| `idle-retire-first-after-a`：提前退休 | 2380.2695 |
| `idle-retire-first-before-a`：恢复基线 | 2251.3110 |

候选比恢复基线慢约5.73%。两个基线接近，结合下述阶段证据，停止扩样并撤回。
不因候选正确性通过或局部Resolve提前而保留负收益改动。

同布局L4对照：

| 指标 | 基线 | 候选 |
| ---- | ---- | ---- |
| 最后贡献组全release→C确认，中位 | 6.267us | 4.435us |
| 全task末kernel→DONE，中位 | 12.512us | 10.4325us |
| 同task退休→下一次claim，中位 | 1.657us | 5.433us |
| publish→全三lane arrive，中位 | 5.407us | 6.553us |
| C/V入口偏斜，中位 | 2.886us | 4.541us |
| t23同cluster相邻claim周期，中位 | 40.534us | 45.477us |
| 全局无kernel间隙 | 921.875us | 1011.268us |
| 其中同task间隙 | 104.706us | 175.637us |
| 首kernel到末kernel窗口 | 2711.793us | 2866.566us |

退休段确实缩短，但discovery被插到退休与同task下一次claim/publish之间，
使连续多block任务新增组周转停顿。同task间隙增加70.931us，占全局空窗增量的大部分。
这是阶段证据支持的解释，不把插桩差值直接当作无插桩Device的精确成本拆分。

## 撤回与重开条件

生产与测试补丁保存在工作区 `reports/csa_poursoul_20261027.Osnaam/` 的
`idle-retire-first-production.patch`、`idle-retire-first-tests.patch`。
主仓恢复原退休/发现顺序及旧策略断言，保留独立的Mix前缀早发和错误门禁测试。
原始结果位于该目录的 `artifacts/`；L4分别是
`idle-prefix8-before-retire-l4`、`idle-retire-first-after-l4`。
候选原生二进制保存在 `runtime-idle-retire-first/`，不覆盖基线归档。

只有能保持连续组的retire→next-claim/publish邻接，或有证据证明不再影响后续组周转时，
才重开这一排序方案。下一候选是复用本轮poll已经获得的空Ready结果，省掉重复discovery，
而不是额外加一次判空，也不缓存跨轮许可；它须独立验证，不能直接沿用本次局部收益。
