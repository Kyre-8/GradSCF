"""An orbital permutation within a fixed occupation sector is a gauge choice."""
import jax.numpy as jnp
import numpy as np
import importlib

from gradscf.gw.g0w0 import g0w0_cd_restricted


def test_fixed_gw_self_energy_is_covariant_under_occupied_orbital_permutation(monkeypatch):
    energy = jnp.array([-3.8, -.3, .9, 3.9])
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(2, 4, 4))*.08
    factors = jnp.asarray((raw+raw.swapaxes(1, 2))/2)
    module = importlib.import_module('gradscf.gw.g0w0')
    original_loop = module._qp_loop
    chemical_potentials = []
    def record_chemical_potential(**arguments):
        chemical_potentials.append(float(arguments['ef']))
        return original_loop(**arguments)
    monkeypatch.setattr(module, '_qp_loop', record_chemical_potential)
    kwargs = dict(nocc=2, df_factors=factors, nw=16,
                  density_matrix=jnp.diag(jnp.array([2., 2., 0., 0.])),
                  fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(energy),
                  evaluate_only=True)
    original = g0w0_cd_restricted(mo_energy=energy, mo_coeff=jnp.eye(4), **kwargs)
    order = np.array([1, 0, 2, 3])
    changed = g0w0_cd_restricted(mo_energy=energy[order], mo_coeff=jnp.eye(4)[:, order], **kwargs)
    np.testing.assert_allclose(changed.sigma_qp, np.asarray(original.sigma_qp)[order],
                               rtol=0, atol=1e-11)
    for chemical in chemical_potentials:
        assert float(jnp.max(energy[:2])) < chemical < float(jnp.min(energy[2:]))
