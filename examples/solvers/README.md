# Shared solver examples

Run from the repository root with `PYTHONPATH=src`. The numerical examples use JAX
float64. The basic examples use small dimensionless matrices; the larger
subspace benchmark and native benzene example add controlled backward checks.

- [shared_solvers.py](shared_solvers.py): a matrix-free eigenvector response
  and a nonsymmetric linear solve.
- [degenerate_subspace.py](degenerate_subspace.py): an internally degenerate
  spectral projector and its first-order response, with a 61-point crossing
  scan and independent dense differences. [plot_degenerate_subspace.py](plot_degenerate_subspace.py)
  plots saved data. See [A03 results and validity boundaries](../../reproducibility/ad_showcase/A03_degenerate_subspace/README.md).

- [nonhermitian_subspace.py](nonhermitian_subspace.py): a nonnormal real matrix
  with an exact internal crossing. Shared Davidson and coupled implicit block
  solves give projector/cluster-trace JVP, VJP and HVP. Raw eigenvectors are
  stopped diagnostics; individual sorted roots remain guarded.

- [degenerate_vjp.py](degenerate_vjp.py): n=256/1024/4096 complete-subspace
  VJPs with analytic references, internal/external gap scans and frame checks.
  [compare_degenerate_eigh_ad.py](compare_degenerate_eigh_ad.py) compares ordinary
  `eigh` AD on the same smooth projector loss. Exact-degeneracy NaNs are recorded
  separately from genuinely undefined cut-cluster derivatives.
- [degenerate_benzene_vjp.py](degenerate_benzene_vjp.py): GradSCF native
  RHF/Cartesian 6-31G* and a 1701-dimensional CIS operator; the complete bright
  doublet has a nonzero invariant transition-tensor VJP. The
  [independent PySCF comparison](compare_degenerate_benzene_vjp.py) uses its own
  SCF/integrals and projector differences. This is fixed-reference source-potential
  response, not relaxed nuclear response. [Results and plots](../../reproducibility/ad_showcase/A03_degenerate_vjp/README.md).

- [ch4_cis_cluster.py](ch4_cis_cluster.py): complete geometry response of a
  bright, triply degenerate singlet CIS cluster with native HF relaxation,
  moving integrals and dipoles. The 15-coordinate VJP, independent PySCF
  directional differences and occupied/virtual frame check are recorded in
  [the CH4 capsule](../../reproducibility/ad_showcase/A03_ch4_cluster_geometry/README.md).

For the CH4 example, run each stage in a bounded process from the repository
root with `PYTHONPATH=src:.`. Use fresh result directories to preserve earlier
evidence:

```python
from pathlib import Path
from examples.solvers.ch4_cis_cluster import run, check_frame
from examples.solvers.compare_ch4_cis_cluster import run as compare, plot

output = Path("reproducibility/ad_showcase/my_ch4_cluster")
run("forward", output=output)
run("response", output=output)
compare(output)
check_frame(output, max_subspace=90)
plot(output)
```

See the [solver API](../../src/gradscf/solvers/README.md) and
[degeneracy contract](../../src/gradscf/solvers/DEGENERACY.md) for supported
observables and differentiation modes.
