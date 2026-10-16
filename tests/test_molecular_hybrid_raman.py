"""Small hybrid/RSH mixed RFF gate before the water geometry/spectra examples."""
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("xc", ["pbe0", "b3lyp", "wb97xd"])
def test_hybrid_raman_direction_matches_relaxed_differences(xc):
    from gradscf import dft, gto
    from gradscf_tools.molecular_response import MolecularEnergy

    start = perf_counter()
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc=xc, grids_level=0, integral_backend="native", max_cycle=200,
                 conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    energy = MolecularEnergy(mf)
    coordinates, field = energy.coordinates, jnp.array([0., 0., .001])
    alpha = jax.jit(jax.jacfwd(energy.dipole, argnums=1))
    tensor = np.asarray(alpha(coordinates, field))
    direct = np.asarray(-jax.jit(jax.hessian(energy, argnums=1))(coordinates, field))
    np.testing.assert_allclose(tensor, direct, atol=2e-7, rtol=1e-7)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    component = lambda r: alpha(r, field)[2, 2]
    mixed = float(jax.jit(lambda r: jax.jvp(component, (r,), (direction,))[1])(coordinates))
    assert np.isfinite(mixed) and abs(mixed) > .1
    errors = []
    for step in (3e-4, 1e-4):
        numerical = float((component(coordinates+step*direction)
                           - component(coordinates-step*direction))/(2*step))
        np.testing.assert_allclose(mixed, numerical, atol=3e-6, rtol=1e-5)
        errors.append(abs(mixed-numerical))
    print(xc, "RFF", mixed, "FD errors", errors,
          "alpha identity error", float(np.max(np.abs(tensor-direct))),
          "seconds", perf_counter()-start, flush=True)
