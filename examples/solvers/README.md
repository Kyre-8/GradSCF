# Shared solver examples

Run from the repository root with `PYTHONPATH=src`. Both examples use JAX
float64 and small dimensionless matrices, without molecular integrals.

- [shared_solvers.py](shared_solvers.py): a matrix-free eigenvector response
  and a nonsymmetric linear solve.
- [degenerate_subspace.py](degenerate_subspace.py): an internally degenerate
  spectral projector and its first-order response.

See the [solver API](../../src/gradscf/solvers/README.md) and
[degeneracy contract](../../src/gradscf/solvers/DEGENERACY.md) for supported
observables and differentiation modes.
