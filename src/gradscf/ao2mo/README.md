# AO-to-MO transformations

`gradscf.ao2mo` owns the shared molecular transformations used by MP, CI, CC,
FCI and response methods. Integral generation and native coordinate products
remain in `gradscf.integrals`.

| Module | Responsibility |
| --- | --- |
| `blocks` | Requested dense, packed, DF or native-direct MO blocks |
| `native` | Bounded shell slabs and two-stage half transformations |
| `reference` | Explicit full MO reference tensors and validation |
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

Native integrals use float64. MO-coefficient and first/second coordinate products
remain live; unsupported primitive basis-parameter and third coordinate products
raise explicitly. On CUDA, native slabs use bounded CPU callbacks and return
device-resident results. This is a compatibility path, not a GPU integral kernel.
