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
             max_memory=60000.).run()
td = tdscf.TDDFT(mf, nstates=3, response_kernel_options={
    "integral_cache": "none",
})
result = td.kernel()
```

## Controls and storage

- `max_memory` defaults to 60000 decimal MB (60 GB). SCF and TDDFT use the
  same requested total budget. TDDFT inherits it from SCF/reference unless a
  TD argument or explicit response-options budget overrides it.
- Planning reserves 20% for runtime/model temporaries and charges known
  resident arrays before choosing workspaces. Device-capacity estimates can
  reduce the effective budget; they do not change the requested value or
  invalidate a converged SCF reference when free memory changes.
- `grid_block_size=0` (default) chooses dense AO only when its primitive
  generation workspace also fits. Otherwise it selects streamed AO blocks.
- A positive `grid_block_size` forces streaming and caps its block length.
  The AO workspace estimate may choose a smaller block.
- `grid_block_size=-1` retains the dense small-system reference path.
- `grid_max_memory=None` (default) derives the grid workspace from the total
  budget. An explicit positive value adds an AO-workspace cap within that
  budget. Primitive intermediates, grid/basis arrays, SCF matrix histories
  and retained input-cache arrays are included in the estimate.
- Automatic SCF blocks have practical ceilings of 4096 points on CPU and
  16384 on GPU; explicit block requests can use up to 16384 when they fit.
  Automatic response blocks
  have an 8192-point ceiling and share space with vector microbatches. These
  practical ceilings avoid increasing temporaries solely to fill unused RAM.
  An adaptive supplier permits the response stage to replan its own width.
  Explicit small block/workspace limits and legacy suppliers retain their cap.

These are working-array estimates, not an enforced process RSS or device
allocator limit. Arbitrary external network activations and XLA/allocator
residency still require a runtime monitor or job limit. The library neither
changes global allocator settings nor initializes unrelated GPU devices.
Eager feature preparation inside the legacy `NeuralXCModel` response binder
is also opaque model workspace. The generic `dft.custom_xc` response binding
is lightweight and uses the planned provider for density/kernel evaluation.

Streaming integral inputs/references have zero-row `ao` and `ao_deriv1` arrays
and a dynamic `grid_ao` supplier. Full quadrature coordinates and weights remain
available in their existing fields. Consumers must use the supplier instead of
treating the zero-row arrays as an empty physical grid. The small-system dense
array interface is unchanged.

The process-wide input cache has both a 32-entry limit and a 256,000,000-byte
limit for cached quadrature/AO arrays. Oversized entries are used for their
current calculation without entering that cache. HDF5 stores the supplier's
compact basis parameters, requested total budget and adaptive-block metadata,
and reuses the reference grid on reload. Legacy files default to 60000 MB and
retain their original block cap. Full AO arrays are not regenerated merely
to save a reference.

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
