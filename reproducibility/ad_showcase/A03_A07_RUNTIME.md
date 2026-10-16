# A03–A07 execution record

The current checks ran on `jy_gpu_1` (`c20`, Intel Xeon Platinum 8457C), CPU
float64, in `/home/yjiao/gradscf_checks/ad-showcase-a03-a07-20261015`.
This separate snapshot copies the frozen source used by A01/A02. Its Python
tree SHA256 is `ca37f375d0e4cc673d1ca2e5b50f189b9047ad8bce56e00f7f397156fac3e43f`;
the native library SHA256 is
`e02844ce6c1b5eebc2dc00a6eec5f1803fd33ad4150aa82e9858d5c0890b9904`.
No production source changed for A03–A07. Each result directory includes the
source manifest and a verification manifest of executed scripts and artifacts.

Python 3.11.15, JAX 0.10.0, NumPy 2.4.4, SciPy 1.17.1, PySCF 2.13.0 and
jax-xc 0.0.9 were installed. Run from the snapshot root:

```bash
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
PYTHON=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
timeout 900 "$PYTHON" examples/solvers/degenerate_subspace.py
timeout 300 "$PYTHON" examples/solvers/plot_degenerate_subspace.py
timeout 900 "$PYTHON" examples/mp/taylor_series.py
timeout 900 "$PYTHON" examples/mp/compare_water_pyscf.py
timeout 1800 "$PYTHON" examples/basis/optimize_exponents.py
timeout 900 "$PYTHON" examples/basis/compare_exponents_pyscf.py
timeout 300 "$PYTHON" examples/basis/plot_exponents.py
timeout 1800 "$PYTHON" examples/localization/archive_ethylene_boys.py
timeout 300 "$PYTHON" examples/localization/plot_ethylene_boys.py
timeout 1800 "$PYTHON" examples/training/external_functional.py
timeout 1800 "$PYTHON" examples/training/dfpt_response.py
timeout 1800 "$PYTHON" -m pytest -q tests/test_external_neural_xc.py \
  -k 'dfpt_functional_parameter_gradient_includes_relaxed_reference or response_loss_training_gradient_and_accepted_step or three_training_modes_have_fd_gradients_and_accept_external_params or parameter_gradient_includes_self_consistent_density_response'
```

Every command above exited zero. The focused A07 group reported **8 passed,
17 deselected in 198.43 s**, with no skips. These commands are validation
records, not an instruction to launch every job simultaneously. The scripts
have editable settings and no CLI parser. Numerical work and plot generation
were remote; local verification used static syntax, whitespace and image checks.

The additional full `examples/mp/he2_convergence.py` attempt, bounded by 1800 s,
exited **134** during its extremum searches with an LLVM allocation failure.
Its failure log is retained under A04. The historical MP30 report and figures
remain historical evidence; neither partial printed rows nor the smaller
current checks establish a successful current MP30 rerun. No shared JIT/cache
implementation was changed in response to that failure.

Elapsed times reported inside scripts include their stated setup/compilation
work and possible shared-node activity. They are not portable performance
comparisons. Trailing measured-result comments added locally after execution
are not part of the executed script hashes; the remote execution versions are
preserved in this snapshot.
