# AO-to-MO transformations

`gradscf.ao2mo` owns the shared molecular transformations used by MP, CI, CC,
FCI and response methods. Integral generation and native coordinate products
remain in `gradscf.integrals`.

| Module | Responsibility |
| --- | --- |
| `blocks` | Requested dense, packed, DF or native-direct MO blocks |
| `df` | Three-index selected DF factors with auxiliary-axis workspaces |
| `native` | Bounded shell slabs and two-stage half transformations |
| `reference` | Explicit full MO reference tensors and validation |
| `frame` | Pure-JAX occupied/virtual projector frame for fixed orbital selections |
| `spin` | Frozen orbital selections and reference spin conversion |
| `response` | Existing occupied/virtual response slices |

The package exports functions directly:

```python
from gradscf import ao2mo

c_occ = mo_coeff[:, :nocc]
c_vir = mo_coeff[:, nocc:]
ovov = ao2mo.transform_eri_block(
    (c_occ, c_vir, c_occ, c_vir),
    eri_pair_matrix=eri_pair_matrix,
    max_memory=60000,
)
```

Supply exactly one of `eri`, `eri_pair_matrix`, `df_factors` or `direct_basis`.
Four different real coefficient spaces are supported. `transform_integrals`
and `transform_unrestricted_integrals` explicitly return complete MO tensors;
their required output storage remains quartic in the MO dimension.

`max_memory` is a forward-array planning budget in decimal MB, default 60000.
Known inputs, output versions and live transformation buffers share the budget
after a runtime reserve. DF uses auxiliary tiles; native-direct transforms use
whole-shell tiles with compact JAX loops and recomputation during AD. Native
projections and dtype promotion are budgeted before allocating their matrices.
Caller-owned solver/AD histories and process RSS need separate accounting.
`transform_df_factors(B, C_occ, C_vir)` keeps the required complete Qia output
and tiles only its auxiliary workspace; it does not form a four-index tensor
or assume symmetric raw factors. Other live owners use `reserved_bytes`.

Native integrals use float64. MO-coefficient and first/second coordinate products
remain live; unsupported primitive basis-parameter and third coordinate products
raise explicitly. On CUDA, native slabs use bounded CPU callbacks and return
device-resident results. This is a compatibility path, not a GPU integral kernel.

`frame.project_frame(C, A0, S, nocc)` projects each fixed initial occupied and
virtual anchor block into the current stationary subspace, then orthonormalizes
it with Cholesky. It defines a local, differentiable frame for the new molecular
CI/FCI frozen/core/active geometry adapter. It preserves the occupied density,
without differentiating arbitrary optimizer recentering rotations. This chosen
gauge need not equal that arbitrary frame away from the anchor geometry; a
rank-losing projection remains nonfinite instead of switching gauges. No SCF,
eigenvalue or orbital-optimization solver is added by this utility.
Selected orbitals follow this anchor convention, not an automatically
recanonicalized core or virtual energy window.
