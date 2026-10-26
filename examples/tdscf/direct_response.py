"""Non-DF benzene TDA: exact integral reuse within a response budget.

Use the same orbital space and five roots for a storage comparison. The
explicit 40-column Davidson space also permits running this example against
the previous implementation to isolate the integral/response storage change.
No geometry derivative is requested from the shell-direct integral engine.
"""
from time import perf_counter
import gc
import json

import jax
import jax.numpy as jnp
import numpy as np

import gradscf
from gradscf import gto, scf, tdscf
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft.response import build_restricted_tda_operator

jax.config.update("jax_enable_x64", True)

# Planar benzene: C-C = 1.397 Angstrom, C-H = 1.083 Angstrom.
atoms = []
for symbol, radius in (("C", 1.397), ("H", 2.480)):
    for angle in np.arange(6) * np.pi / 3:
        atoms.append((symbol, (radius*np.cos(angle), radius*np.sin(angle), 0.)))
mol = gto.M(atom=atoms, basis="6-31g*", unit="Angstrom", cart=True)

started = perf_counter()
mf = scf.RHF(mol, integral_backend="native", jk_backend="direct", grids_level=0,
             conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9,
             max_cycle=200).run()
scf_seconds = perf_counter() - started
assert mf.converged

started = perf_counter()
reference = scf.as_reference(mf)
reference_seconds = perf_counter() - started
eri_bytes = sum(0 if getattr(reference, name) is None else getattr(reference, name).nbytes
                for name in ("rep_tensor", "eri_pair_matrix", "df_factors"))
action, _, gaps = build_restricted_tda_operator(reference, SemilocalResponseFunctional("hf"))
plan = getattr(action, "memory_plan", None)
response_cache_bytes = 0 if plan is None else plan.eri_cache_bytes
vectors = jnp.asarray(np.random.default_rng(910).normal(size=(5, gaps.size)))
started = perf_counter()
compiled = jax.jit(action).lower(vectors).compile()
compile_seconds = perf_counter() - started
compiled(vectors).block_until_ready()
seconds = []
for _ in range(3):
    started = perf_counter()
    compiled(vectors).block_until_ready()
    seconds.append(perf_counter() - started)
memory = compiled.memory_analysis()
memory_bytes = {name: getattr(memory, name) for name in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                 "alias_size_in_bytes", "const_size_in_bytes") if hasattr(memory, name)}

# The operator benchmark and the full solve have separate compiled lifetimes.
del compiled, action, vectors
gc.collect()
jax.clear_caches()
started = perf_counter()
td = tdscf.TDA(mf, nstates=5, davidson_tol=1e-8, davidson_max_iter=100,
               davidson_max_subspace=40)
result = td.kernel()
solve_seconds = perf_counter() - started
assert bool(result.converged)
print("DIRECT_RESPONSE_RESULT " + json.dumps(dict(
    molecule="benzene", basis="6-31g*", cart=True,
    source_module_file=gradscf.__file__,
    nao=mf.mo_coeff.shape[0], transitions=int(gaps.size), nroots=5,
    max_subspace=40, dtype="float64", backend=jax.default_backend(),
    hf_energy_hartree=float(mf.e_tot), excitation_energy_hartree=np.asarray(td.e).tolist(),
    eri_cache_bytes=int(eri_bytes), response_eri_cache_bytes=int(response_cache_bytes),
    response_batch_size=None if plan is None else plan.batch_size,
    scf_seconds=scf_seconds,
    reference_seconds=reference_seconds, operator_compile_seconds=compile_seconds,
    warm_action_median_seconds=float(np.median(seconds)), full_tda_seconds=solve_seconds,
    compiled_memory_bytes=memory_bytes), sort_keys=True))
