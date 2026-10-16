"""G07: a smooth constrained RHF surface and loss of spin stability in H2.

Differentiate the converged RHF branch, not the discrete choice between RHF
and broken-symmetry UHF. A negative spin curvature changes which reference
is stable; it does not make the constrained RHF energy curve discontinuous.
"""
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.molecular_response import MolecularEnergy


basis = "6-31g*"
lengths = np.linspace(1., 5.8, 13)  # Bohr; fixed restricted occupation topology
controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                integral_backend="native", max_cycle=200)
seed = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 1.4", basis=basis, unit="Bohr", cart=True),
               **controls).run()
energy = MolecularEnergy(seed, gradient_tolerance=1e-10)
overlap_engine = NativeRHF(seed.mol.to_spec(), basis=basis)
coordinates = lambda r: jnp.array([[0., 0., 0.], [0., 0., r]])
surface = lambda r: energy(coordinates(r))
value_gradient = jax.jit(jax.value_and_grad(surface))
curvature = jax.jit(jax.grad(jax.grad(surface)))
state_at = jax.jit(lambda r: energy.solve(coordinates(r)))
output = Path("reproducibility/ad_showcase/G07_h2_surface")
output.mkdir(parents=True, exist_ok=True)
rows = []
previous_coordinates = previous_occupied = None
start = perf_counter()
for length in lengths:
    value, first = map(float, value_gradient(length))
    second = float(curvature(length))
    state = state_at(length)
    if not bool(state.stationary) or not np.isfinite([value, first, second]).all():
        raise RuntimeError(f"RHF response failed at R={length} Bohr.")
    occupied = np.asarray(state.mo_coeff[:, :1])
    current_coordinates = np.asarray(coordinates(length))
    overlap = 1.
    if previous_coordinates is not None:
        cross = overlap_engine.cross_overlap(previous_coordinates, current_coordinates)
        overlap = float(np.abs(previous_occupied.T @ np.asarray(cross) @ occupied).item())
        assert overlap > .9
    previous_coordinates, previous_occupied = current_coordinates, occupied

    checks = []
    for step in (1e-3, 3e-4):
        plus, dplus = map(float, value_gradient(length+step))
        minus, dminus = map(float, value_gradient(length-step))
        checks.append(dict(step_bohr=step, gradient_error=abs((plus-minus)/(2*step)-first),
                           curvature_error=abs((dplus-dminus)/(2*step)-second)))
    assert checks[-1]["gradient_error"] < 2e-6
    assert checks[-1]["curvature_error"] < 2e-6

    # The public stability/following API owns the spin-channel Hessian and
    # branch selection. Its result is a separate primal diagnostic, not AD.
    mol = gto.M(atom=list(zip(["H", "H"], current_coordinates)), basis=basis, unit="Bohr", cart=True)
    restricted = scf.RHF(mol, **controls).run()
    np.testing.assert_allclose(value, restricted.e_tot, atol=1e-9, rtol=0)
    spin = restricted.stability(channel="spin", solver="dense", eigensolver_tol=1e-10)
    assert spin.stationary and spin.eigensolver_converged and spin.spectrum_certified
    selected = restricted.stabilize(channel="spin", solver="dense", max_restarts=5,
                                     step_sizes=(.5, 1.))
    assert selected.stable and selected.result.converged
    coefficients = np.asarray(selected.result.mo_coeff)
    spin_square = 0.
    if coefficients.ndim == 3:
        metric = np.asarray(overlap_engine.cross_overlap(current_coordinates, current_coordinates))
        # N_alpha=N_beta=1: <S^2> = 1 - |<phi_alpha|phi_beta>|^2.
        spin_square = 1-float(np.abs(coefficients[0, :, :1].T @ metric @ coefficients[1, :, :1]).item())**2
    row = dict(bond_bohr=float(length), rhf_energy_hartree=value,
               gradient_hartree_per_bohr=first, curvature_hartree_per_bohr2=second,
               occupied_overlap_previous=overlap, finite_difference_checks=checks,
               spin_minimum_curvature=spin.minimum_curvature,
               spin_stable=bool(spin.stable), selected_energy_hartree=float(selected.result.e_tot),
               selected_model=type(selected.result).__name__, spin_square=spin_square,
               selection_status=selected.status, selection_restarts=selected.restarts)
    rows.append(row)
    print(f"R={length:.2f}: RHF E={value:.10f}, dE={first:.5f}, d2E={second:.5f}, "
          f"spin curvature={spin.minimum_curvature:.5f}, selected={row['selected_model']}", flush=True)
    (output / "gradscf.json").write_text(json.dumps(dict(
        basis=basis, cart=True, records=rows, elapsed_seconds=perf_counter()-start,
        hostname=platform.node(), jax_version=jax.__version__, dtype="float64",
        differentiation="implicit on fixed RHF branch; selection is not differentiated"), indent=2)+"\n")

# Measured on jy_gpu_1, CPU float64, 2026-10-15:
# R=2.2 Bohr: E_RHF=-1.0629274521 Eh; spin curvature=+0.04945 Eh; RHF selected.
# R=2.6 Bohr: E_RHF=-1.0209781162 Eh; spin curvature=-0.29660 Eh; UHF selected.
# Independent PySCF max gradient/Hessian errors: 7.258e-13 / 1.253e-13 au.
# At R=5.8 Bohr, selected unrestricted <S^2>=0.99629311 (spin contaminated).
# Runtime including all native checks/compilation: 142.38 s.
