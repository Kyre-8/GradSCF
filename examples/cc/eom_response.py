"""First-order EE/IP/EA energy response to an MO-basis one-electron perturbation.

The MO frame is fixed. CC amplitudes are solved inside the differentiated
function, retaining their response. This is not a nuclear-coordinate gradient.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc

mf = scf.RHF(
    gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", unit="Angstrom"),
    conv_tol=1e-12,
).run()
reference = cc.CCSD(mf).run().reference
h, g = reference.h1, reference.eri
perturbation = jnp.array([[0.2, 0.04], [0.04, -0.1]])
ground_config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)

for sector in ("ee", "ip", "ea"):
    config = cc.EOMConfig(sector=sector)

    def energy(t):
        ht = h + t * perturbation
        ground = cc.run_cc(ht, g, nocc=1, config=ground_config)
        return cc.run_eom(
            ht,
            g,
            ground,
            nocc=1,
            config=config,
            cc_config=ground_config,
        ).energies[0]

    value, derivative = jax.jit(jax.value_and_grad(energy))(0.0)
    step = 1e-4
    finite_difference = (energy(step) - energy(-step)) / (2 * step)
    print(sector.upper(), "energy:", float(value), "Hartree")
    print("  d omega / dt:", float(derivative))
    print("  central difference:", float(finite_difference))
    print("  absolute error:", float(jnp.abs(derivative - finite_difference)))
