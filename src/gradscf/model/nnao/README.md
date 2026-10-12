# NNAO 模型模块

NNAO 使用 MACE 预测收缩系数，并通过 GradSCF 的基组组装、native 积分和
implicit SCF 计算变分能量与梯度。此目录包含 GradSCF 的实现；原始 MACE
项目隔离在 [`vendor/mace-jax/`](vendor/mace-jax/)，由其 `UPSTREAM.json`
记录版本和逐文件哈希，许可证保留在上游目录。

## 文件归属

- `encoder.py`、`graph.py`：网络和输入图。
- `initialization.py`：原有初始指数与参考收缩模板；本次整理不改变数值。
- `basis.py`、`grimme.py`、`data/`：基组组装和物理资产。
- `integrals/context.py`：orbital／auxiliary 计划与固定 primitive 数据准备。
- `integrals/fixed_exponents.py`：固定指数下的 JAX 收缩及其自动微分。
- `integrals/variable_exponents.py`：native contracted 积分及系数／指数 VJP。
- `integrals/cache.py`：primitive 缓存身份、NPZ 与只读 mmap。
- `training/objective.py`：`BasisRHF` 能量与 SCF 有效性诊断。
- `training/trainer.py`、`checkpoints.py`：接受状态、优化器与续训。
- `training/alternating.py`、`native.py`：现有两种指数更新实现。
- `training/validation.py`：数值检查；`runtime.py`：数据读取与来源校验。
- `workflows/`：固定指数训练、交替标定、native 联合训练及检查。

## 运行

在已有依赖环境中，从仓库根目录运行。使用本地上游快照时：

```bash
export PYTHONPATH="$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax"
python -m gradscf.model.nnao.workflows.validate --help
python -m gradscf.model.nnao.workflows.train --help
python -m gradscf.model.nnao.workflows.joint --help
python -m gradscf.model.nnao.workflows.joint_ad --help
python -m gradscf.model.nnao.workflows.optimize --help
```

生产共享训练的来源校验要求 MACE 来自本地快照。公共模型的可选安装入口
仍为 `GradSCF[nnao]`；源码快照不作为嵌套 Python 包打进 GradSCF wheel。
`tools/train_nnao_*.py` 等保留原命令行参数，仅调用这些模型内工作流。

训练主体可直接导入，不依赖仓库 `tools/`：

```python
from gradscf.model.nnao.training import BasisRHF, NativeJointEngine, JointAdam
from gradscf.model.nnao.integrals import prepare_native_context, native_projection
```

`BasisRHF` 当前限于中性闭壳层 RHF；默认的甲烷几何用于最小演示。
共享训练工作流保留原有 CHNOF、`szp663_direct`/core6 配置及参数顺序。
这次文件重组没有扩大已验证的科学适用范围。

## 积分与微分约定

固定几何／指数时，缓存 primitive 的 S、h 和 packed DF 因子；JAX 通过
含归一化的收缩矩阵投影到 contracted AO。DF 投影按辅助指标分块展开，
但当前实现仍需要完整 packed 输入与完整 contracted 三指标输出，不是磁盘流式算法。

native 联合路径直接计算 contracted S、h、三中心积分并求系数／指数 VJP。
辅助基度量在固定几何和辅助基时复用。不缓存完整四指标 ERI。
当前 DF 参数 AD 不包含几何／辅助基导数，不能据此宣称完整力训练已受支持。

本次迁移保留现有数值缓存签名与严格来源检查：旧的已验证缓存不被静默
解释为新版本训练结果。完整训练源码身份因迁移变化，旧运行目录不得直接
作为同一来源续训；需显式恢复参数并重新验证，不移除来源保护。

初始指数重新标定、低频指数的解析梯度交替更新，以及更大规模缓存管理
是后续工作，本次重组没有将它们冒充为已完成的训练功能。

更多已记录实验和方法说明见 [NNAO.md](NNAO.md)。
