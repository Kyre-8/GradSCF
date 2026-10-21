# Molecular AO grid streaming

Molecular RKS/UKS can integrate XC without keeping full-grid AO values and
spatial derivatives. The same AO supplier is retained by the response
reference and used by TDDFT/UTDDFT. Exact exchange continues to use the selected
J/K backend; streaming does not change quadrature, the functional, occupations,
or the excitation space.

```python
from gradscf import dft, gto, tdscf

mol = gto.M(atom="O 0 0 0; H 0 .76 .59; H 0 -.76 .59",
            basis="def2-svp", cart=True)
mf = dft.RKS(mol, xc="pbe", jk_backend="direct", grids_level=3,
             grid_max_memory=256.).run()
td = tdscf.TDDFT(mf, nstates=3, response_kernel_options={
    "max_memory": 2000., "integral_cache": "none", "grid_block_size": 256,
})
result = td.kernel()
```

## Controls and storage

- `grid_block_size=0` (default) selects streaming when the estimated resident
  AO arrays would consume more than one quarter of `grid_max_memory`.
- A positive `grid_block_size` forces streaming and caps its block length.
  The AO workspace estimate may choose a smaller block.
- `grid_block_size=-1` retains the dense small-system reference path.
- `grid_max_memory` is a decimal-MB estimate for AO evaluation work arrays,
  including padded primitive intermediates. It is not a process RSS or device
  allocator limit. SCF matrices, J/K, model internals and Davidson have their
  own storage. Response blocks respect both their response budget and the AO
  supplier's block limit.

Streaming integral inputs/references have zero-row `ao` and `ao_deriv1` arrays
and a dynamic `grid_ao` supplier. Full quadrature coordinates and weights remain
available in their existing fields. Consumers must use the supplier instead of
treating the zero-row arrays as an empty physical grid. The small-system dense
array interface is unchanged.

The process-wide input cache has both a 32-entry limit and a 256,000,000-byte
limit for cached quadrature/AO arrays. Oversized entries are used for their
current calculation without entering that cache. HDF5 stores the supplier's
compact basis parameters and reuses the reference grid on reload; full AO
arrays are not regenerated merely to save a reference.

## Automatic differentiation and external models

AO centers, exponents, contraction coefficients, quadrature coordinates and
weights are dynamic PyTree leaves. A checkpoint encloses AO evaluation and
the consuming block contraction together, so reverse AD can recompute the
large block intermediates. The final partial block has zero quadrature weights;
small feature maps remove padded points before an external model is called.
No geometry dependency is stopped to implement streaming.

`dft.custom_xc` still receives the complete externally defined input PyTree.
Density, density gradients, kinetic density and density Laplacian are generated
as small full-grid features. Global operations such as the square of an
integrated network output retain their cross-grid coupling. Fixed-density,
explicit and implicit training retain their existing definitions.

An external `DensityInputs.ao` or `ao_deriv1` request explicitly materializes
the corresponding full-grid array for compatibility. Such a model does not
have the same memory bound as one using density features. Legacy local-HFX
and PT2 feature-cache builders still require dense AO data and explicitly
reject streaming; ordinary hybrid exact exchange does not use those caches.
Noncollinear GKS retains its existing dense implementation and rejects a
forced streaming request.

The provider uses JAX array operations and participates in device transfer.
CPU numerical verification does not establish GPU runtime performance.

## Regression coverage

The `test_streamed_*` tests compare dense and streamed AO values, XC energies,
potentials, response actions, external global networks, cache roundtrips and
public SCF/TDDFT workflows. They include coordinate/basis JVP/VJP/HVP and
moving-grid relaxed excitation gradients. Structural JAXPR checks distinguish
bounded AO representation from actual measured peak RSS.
