# A03 — A whole eigenspace through an internal crossing

The extended [existing example](../../../examples/solvers/degenerate_subspace.py)
passed its 61-point scan on `jy_gpu_1` in 176.81 s. It uses the shared Hermitian
solver and subspace-response interface. No replacement eigensolver or response
algorithm is introduced. [Figure](crossing_response.png), [PDF](crossing_response.pdf),
[numeric data](results.json), [run log](../../../artifacts/ad_showcase/provenance/A03_degenerate_subspace/A03_native.log), and
[execution environment](../A03_A07_RUNTIME.md) are archived.

The toy matrix is `A(t)=diag(1,1,3,5)+t B`, with the real symmetric perturbation
and probe explicitly saved in JSON. Everything is dimensionless. The first two
eigenvalues are exactly degenerate at `t=0`. The selected rank-2 projector `P`
remains isolated from the other two roots: its exterior gap is 2 at the crossing
and at least 1.8819 over `t=-0.15...0.15`.

The observables are the basis-independent projected
probe norm `probeᵀ P probe` and subspace energy `tr(PA)`. At zero their
derivatives are `-0.115` and `-0.1`, respectively. JVP and VJP of the probe
observable agree to roundoff. Across the scan the largest derivative discrepancy
against independent NumPy eigendecomposition central differences at `h=1e-5`
is `1.50e-10`; results at `h=1e-3,1e-4,1e-5` are retained. The independent
oracle forms the projector from the complete pair, without differentiating
NumPy eigenvectors at degeneracy.

Checks also enforce `P²=P`, `P(dP)P=0` and `(dP)P+P(dP)=dP`. A rank-1 request
cuts the exact cluster and returns `response_valid=False`, status 2; its AD
response is explicitly nonfinite. This invalid response is expected and tested.

In contrast, the lowest **sorted** eigenvalue has left derivative `+0.6` and
right derivative `-0.7` at zero. Central differences approach their average,
which is not a unique derivative. The plot therefore leaves a gap at zero in
each sorted-root slope and does not assign an arbitrary eigenvector derivative.
The first-order branch slopes come from the perturbation restricted to the
degenerate pair. They do not select a unique state/gauge at the crossing.

The boundary is an isolated, fixed-rank subspace. This example validates
first-order projector/trace response, not individual-state tracking or every
higher derivative at an exterior-gap closure. See the
[degeneracy contract](../../../src/gradscf/solvers/DEGENERACY.md).
