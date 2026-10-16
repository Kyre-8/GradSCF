"""Original omegaB97X-D moving-integral/grid force through shared SCF AD."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import dft, gto, scf
from gradscf.scf.inputs import build_rks_integral_inputs


def test_native_water_force_matches_relaxed_directional_energy_differences():
    mol = gto.M(atom="O 0 0 0; H 0 .76 .58; H 0 -.76 .58", basis="sto-3g")
    mf = dft.RKS(mol, xc="wb97xd", conv_tol=1e-11, conv_tol_grad=1e-8).run()
    assert mf.converged
    spec = mol.to_spec()
    seed = jnp.asarray(mf.mo_coeff)
    occupations = np.stack([np.asarray(mf.mo_occ)/2]*2)
    config = scf.SCFDifferentiationConfig(mode="implicit", tolerance=1e-10,
                                         max_iter=100, restart=40)

    def solve(coordinates):
        data = build_rks_integral_inputs(atom=replace(spec, coords_bohr=coordinates),
            basis="sto-3g", xc_spec="wb97xd", grids_level=0, include_dipole_integrals=False)
        return scf.minimize_roks_from_integrals(overlap=data.overlap, hcore=data.hcore,
            eri=data.eri, eri_lr=data.eri_lr, dispersion_energy=data.dispersion_energy,
            nuclear_repulsion=data.nuclear_repulsion, ao=data.ao, ao_deriv1=data.ao_deriv1,
            grid_weights=data.grid_weights, xc_spec="wb97xd", mo_coeff=seed, mo_occ=occupations,
            orthonormalize_initial=True, max_iterations=100, gradient_tolerance=2e-9,
            differentiation=config)

    coordinates = jnp.asarray(spec.coords_bohr)
    direction = jnp.array([[.1, -.1, .05], [-.12, .2, -.04], [.02, -.1, -.01]])
    direction /= jnp.linalg.norm(direction)
    compiled = jax.jit(solve)
    state = compiled(coordinates)
    assert bool(state.stationary), float(state.gradient_norm)
    np.testing.assert_allclose(state.total_energy, mf.e_tot, atol=3e-9, rtol=0)
    gradient = jax.jit(jax.grad(lambda r: solve(r).total_energy))(coordinates)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)
    observed = jnp.sum(gradient*direction)
    for step in (2e-4, 1e-4):
        plus, minus = compiled(coordinates+step*direction), compiled(coordinates-step*direction)
        assert bool(plus.stationary) and bool(minus.stationary)
        finite = (plus.total_energy-minus.total_energy)/(2*step)
        np.testing.assert_allclose(observed, finite, atol=3e-6, rtol=4e-5)
        print("water wb97xd directional force, step, AD-FD:", -float(observed), step,
              float(observed-finite))
