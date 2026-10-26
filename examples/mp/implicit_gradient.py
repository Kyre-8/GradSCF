"""Differentiate MP2/MP3 through implicit HF response in H2/STO-3G.

Scale one primitive contraction coefficient on the first H atom. Native
primitive integrals are computed once; contractions, implicit HF response,
MO transformation and perturbation energies participate in AD. The derivative
is with respect to a dimensionless basis parameter, not a nuclear coordinate.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, integrals, scf, mp
from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix, contract_integrals
from gradscf.ao2mo import transform_integrals

mol = gto.M(atom="H 0 0 0; H 0 0 .9", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-13, conv_tol_grad=1e-11).run()
spec = mol.to_spec()
topology, parameters = integrals.prepare_basis(spec, "sto-3g")
primitive_topology, primitive_parameters = primitive_basis(topology, parameters)
plan = integrals.make_plan(primitive_topology)
primitive = dict(overlap=plan.evaluate("overlap", primitive_parameters),
                 hcore=plan.evaluate("kinetic", primitive_parameters)
                       + plan.evaluate("nuclear", primitive_parameters),
                 eri=plan.evaluate("eri", primitive_parameters))
occupations = jnp.stack([mf.mo_occ / 2] * 2)


def energy(scale, order):
    coefficients = (parameters.coefficients[0].at[0, 0].multiply(scale),
                    *parameters.coefficients[1:])
    contraction = contraction_matrix(topology, replace(parameters, coefficients=coefficients))
    data = contract_integrals(primitive, contraction)
    result = scf.minimize_roks_from_integrals(
        overlap=data["overlap"], hcore=data["hcore"], eri=data["eri"],
        nuclear_repulsion=spec.nuclear_repulsion,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        mo_coeff=mf.mo_coeff, mo_occ=occupations, xc_spec="hf",
        orthonormalize_initial=True, max_iterations=300, gradient_tolerance=1e-10,
        differentiation=scf.SCFDifferentiationConfig(max_iter=100, restart=40),
    )
    h, g = transform_integrals(data["hcore"], result.mo_coeff, eri=data["eri"])
    pt = mp.run_mp(h, g, nocc=1, nuclear_repulsion=spec.nuclear_repulsion,
                  config=mp.MPConfig(order=order, with_t2=False))
    return jnp.where(result.stationary & pt.valid, pt.total_energy, jnp.nan)


step = 1e-4
records = []
for order in (2, 3):
    objective = lambda x: energy(x, order)
    value, derivative = jax.jit(jax.value_and_grad(objective))(1.)
    finite_difference = (objective(1 + step) - objective(1 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-7, rtol=1e-5)
    records.append((float(value), float(derivative), float(finite_difference)))
    print("MP%d  E = % .12f  AD = % .9f  FD = % .9f" %
          (order, value, derivative, finite_difference))

# Example output (CPU, JAX float64; derivative in Hartree per unit scale):
# MP2  E = -1.109271845527  AD =  0.025838166  FD =  0.025838166
# MP3  E = -1.116349572980  AD =  0.025318695  FD =  0.025318695
