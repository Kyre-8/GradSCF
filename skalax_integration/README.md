# skalax_integration

把 [skalax](https://github.com/Brogis1/skalax) 的神经网络交换关联泛函
（Skala）接入 GradSCF 的可微 SCF，用它替代传统 XC 泛函做自洽场计算。

## 每个文件的作用

| 文件 | 作用 |
| --- | --- |
| `features.py` | 把 GradSCF 的 `RestrictedMolecule` + 总密度矩阵，转成 Skala 需要的特征 dict（`density`/`grad`/`kin`/`grid_coords`/`grid_weights`/`coarse_0_atomic_coords`）。负责两边的约定对齐（闭壳层 α=β=total/2、τ 的 0.5 因子、Bohr 单位）。 |
| `adapter.py` | `SkalaFunctionalAdapter`：把 `SkalaFunctional` 包装成 GradSCF `DifferentiableSCF` 认识的鸭子类型接口（`scf_xc_energy_and_alpha_for_density`、`energy_from_molecule`）。另提供 `load_skala_functional()` 加载自带权重。 |
| `run_skala_scf.py` | 端到端示例：构建分子、用 Skala 跑自洽 DFT、打印 HF 参考能和 Skala-DFT 总能。 |
| `__init__.py` | 包导出。 |

## 工作原理（为什么能接上）

- GradSCF 的 `DifferentiableSCF` 不依赖 functional 的具体实现，只要对象暴露
  `scf_xc_energy_and_alpha_for_density(params, molecule, density) -> (exc, alpha)`
  即可。它会用 `jax.value_and_grad` 对这个标量 `exc` 关于密度矩阵求导来构造
  XC 势矩阵。
- Skala 本身就是纯 JAX/Equinox、可微的模型，`model.get_exc(features)` 返回标量
  Hartree 能量，所以整条链路（密度矩阵 → ρ/∇ρ/τ → Skala → E_xc）可以无缝自动微分。
- Skala 是纯密度泛函（不含精确交换），所以 `alpha = 0`。

## 运行

前置：已建 `.venv`、装了 GradSCF（含原生积分后端）和 `skalax`/`equinox`/
`e3nn-jax`，并已执行 `python -m gradscf.integrals._native.build`。

```bash
cd /Users/a1/GradSCF
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python -m skalax_integration.run_skala_scf --molecule h2
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python -m skalax_integration.run_skala_scf --molecule h2o
```

## 约定与限制

- 目前只支持**闭壳层（restricted）**参考态：自旋密度按 α=β=total/2 拆分。
- 特征单位/约定已与 Skala（libxc/PySCF 风格）对齐：坐标 Bohr，`tau = 0.5 Σ n|∇φ|²`。
- Skala 权重为冻结的预训练权重，`params` 传空字典即可。
- 初始密度来自 `xc_spec="hf"` 的一次 HF 计算，仅作初猜；自洽后能量为 Skala 泛函的能量。
- 只实现了基态能量路径；激发态响应（TDDFT kernel）尚未接入。
