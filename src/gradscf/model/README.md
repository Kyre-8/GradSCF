# 神经网络模型开发流程

每个模型在 `gradscf.model.<name>` 内维护自己的网络、物理接入、训练和
工作流。GradSCF 公共模块提供积分、SCF、响应和微分算子，不维护独立的
顶层 `training` 或 `workflows`。模型之间不相互导入 Trainer。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `config.py`、网络模块 | 结构配置、参数初始化和网络预测 |
| 输入与组装模块 | 单位、特征、对称性、输出的物理约束 |
| `integrals/`（需要时） | 模型专用积分执行、设备放置和缓存，调用公共积分内核 |
| `training/` | 损失、梯度、优化器、接受条件和检查点 |
| `workflows/` | 数据选择、运行配置、训练／评估启动和报告 |
| `data/` | 必需的小型物理资产和来源记录，不存训练输出 |
| `vendor/`（需要时） | 模型依赖的上游快照、许可证和来源，不修改上游文件 |

目录按实际复杂度创建；小模型可以只用 `training.py`，不要求建立空的
通用模板。公共接口由 `__init__.py` 显式导出，可选神经网络依赖延迟加载。

## 新建一个模型

1. **定义物理作用。** 说明预测的是能量、XC 能量密度、基组参数或其他物理量，
   列出单位、形状、元素／壳层顺序、dtype 和适用体系。区分训练参数与固定资产。
2. **实现初始化和前向。** 参数显式传入可微函数；Linen、NNX 或纯 JAX 均可。
   NNX 的图定义、可训练参数和非训练状态分开管理。库导入不选择设备、不修改
   JAX 全局配置、不启动计算、不访问训练文件。
3. **实现物理接入。** XC 使用 `gradscf.dft.Functional` 或现有 XC 协议；
   基组模型生成 `BasisParameters`；几何修正项声明其是否进入 SCF 势。
   使用现有物理组装和归一化算子，避免重复计数。
4. **实现模型自己的积分工作流。** 说明缓存依赖、失效条件、packed 张量约定、
   CPU/GPU 边界和支持的微分阶数。不支持的导数明确失败。不会为训练构造完整 ERI。
5. **定义损失与梯度。** SCF 求导调用既有 implicit／explicit 接口，不复制求解器。
   固定 Hamiltonian 的基组优化可以最小化收敛 SCF 能量；可学习 XC 能量不能直接
   将无约束能量下降当作训练目标。样本权重和归一化必须明确。
6. **实现更新和恢复。** 写清 epoch、batch、尝试更新和接受更新的含义。
   参数组的冻结必须覆盖梯度执行与优化器状态。失败时保留完整的已接受状态，
   失败候选单独保存。模型导出与可精确续训的检查点是不同产物。
7. **提供最小工作流。** 使用 `python -m gradscf.model.<name>.workflows.<task>`；
   `tools/` 可以保留薄 CLI。库模块不得反向加载 `tools/` 中的 Python 文件。

## 验收要求

- 初始化恢复预期物理基线；单位、形状、排列和所需对称性正确。
- 完整损失梯度通过方向有限差分。涉及力／响应时另测相应混合导数，不能只测网络。
- 小体系至少执行一次真实更新；比较重构前后的能量、梯度和下一步优化器结果。
- 保存／恢复保留参数树、非训练状态、优化器、随机状态和数据遍历位置；
  对需要 SCF 初始态才能重现分支的工作流，另外记录或确定性重建这些状态。
- 注入 SCF／非有限梯度失败，确认没有接受候选参数或候选 Adam 状态。
- 测量准备、首次编译、热调用、传输和完整训练步；记录硬件、dtype、种子和精度。
- 测试位于 `tests/`；上游快照测试与 GradSCF 回归分开收集。

## 当前入口

- Neural XC：`gradscf.model.neural_xc.training.Sample/Trainer`；
  实验编排：`gradscf.model.neural_xc.workflows`。外部 `dft.Functional`
  仍可使用该 Trainer，无需导入内置网络实现。
- NNAO：`gradscf.model.nnao.training`、`integrals`、`workflows`；
  详见 [NNAO 指南](nnao/README.md)。当前固定指数、Powell 交替和 native AD
  联合训练是显式选择的不同工作流，整理目录不改变它们的数学行为。
- Neural D：`gradscf.model.neural_d` 提供色散模型和泛函组合；目前没有独立
  Trainer，未添加空训练框架。未来专用训练代码归入其自身模块。
- 通用标量能量的力和混合导数辅助函数位于 `gradscf.scf.forces`。

## 命名迁移

`gradscf.training` → `gradscf.model.neural_xc.training`；
`gradscf.workflows` → `gradscf.model.neural_xc.workflows`。
旧顶层入口已撤下，项目内调用点同时更新，不维护转发包。
