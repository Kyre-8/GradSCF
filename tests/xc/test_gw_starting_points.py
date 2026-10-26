"""XC starting points must supply the complete mean-field potential to GW."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _spin_source():
    energy = jnp.array([[-.6, .9], [-.5, 1.]])
    coeff = jnp.tile(jnp.eye(2), (2, 1, 1))
    occupations = jnp.array([[1., 0.], [1., 0.]])
    density = jnp.tile(jnp.diag(jnp.array([1., 0.])), (2, 1, 1))
    reference = SimpleNamespace(mo_energy=energy, mo_coeff=coeff, mo_occ=occupations,
        rdm1=density, h1e=jnp.diag(jnp.array([-1., 1.5])),
        fock_matrix=jnp.stack([jnp.diag(e) for e in energy]),
        df_factors=jnp.zeros((1, 2, 2)), rep_tensor=None,
        dipole_integrals=jnp.zeros((3, 2, 2)), exact_exchange_fraction=.25)
    return SimpleNamespace(xc='pbe0', mo_energy=energy, mo_coeff=coeff,
        mo_occ=occupations, reference=reference, scf_result=None,
        converged=True, _cached_scf_key=('fixed',), _scf_signature=lambda: ('fixed',))


@pytest.mark.parametrize('method', ['g0w0', 'evgw'])
def test_unrestricted_dft_subtracts_actual_spin_fock_in_zero_coulomb_limit(method):
    from gradscf.gw import UGW

    source = _spin_source()
    result = UGW(source, nw=8, method=method, max_cycle=8, conv_tol=1e-8).run()
    assert result.converged
    np.testing.assert_allclose(result.mo_energy, [[-1., 1.5], [-1., 1.5]], atol=1e-9)


def test_rsh_reference_opt_in_does_not_weaken_other_consumer_guards():
    from gradscf.scf.reference import as_reference, reference_state_signature

    source = _spin_source()
    source.xc = 'wb97xd'
    with pytest.raises(NotImplementedError, match='Range|range'):
        as_reference(source)
    assert as_reference(source, allow_range_separated=True) is source.reference
    reference_state_signature(source, allow_range_separated=True)


def test_uks_reference_carries_actual_spin_fock_as_a_differentiable_leaf():
    from dataclasses import fields
    from gradscf.scf.molecules import UnrestrictedMolecule

    assert 'fock_matrix' in {field.name for field in fields(UnrestrictedMolecule)}


def _restricted_source():
    energy = jnp.array([-.6, .9])
    coeff, occ = jnp.eye(2), jnp.array([2., 0.])
    result = SimpleNamespace(mo_energy=energy, mo_coeff=coeff, mo_occ=occ,
        fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(jnp.array([-1., 1.5])),
        density_matrix=jnp.diag(occ), nuclear_repulsion=0.)
    inputs = SimpleNamespace(df_factors=jnp.zeros((1, 2, 2)), eri_pair_matrix=None,
        dipole_integrals=jnp.zeros((3, 2, 2)))
    return SimpleNamespace(xc='pbe0', mo_energy=energy, mo_coeff=coeff, mo_occ=occ,
        scf_result=result, _scf_inputs=inputs, reference=None, converged=True,
        _cached_scf_key=('fixed',), _scf_signature=lambda: ('fixed',))


def test_qsgw_public_facade_accepts_the_same_complete_ks_source():
    from gradscf.gw import GW

    result = GW(_restricted_source(), method='qsgw', nw=8,
                max_cycle=8, conv_tol=1e-8).run()
    assert result.converged
    np.testing.assert_allclose(result.mo_energy, [-1., 1.5], atol=1e-8)
    snapshot = result.get_bse_inputs()
    np.testing.assert_array_equal(snapshot['qp_computed_mask'], [True, True])
    np.testing.assert_array_equal(snapshot['qp_converged_mask'], [True, True])
    np.testing.assert_allclose(snapshot['screening_energy'], result.mo_energy)


def test_scgw_public_facade_preserves_matsubara_result_and_rejects_inferred_bse():
    from gradscf.gw import GW

    result = GW(_restricted_source(), method='scgw', nw=16, beta=20.,
                max_cycle=8, conv_tol=1e-7).run()
    assert result.converged
    assert result.mo_energy is None
    np.testing.assert_allclose(result.result.self_energy_iw, 0., atol=1e-12)
    with pytest.raises(NotImplementedError, match='QP|continuation'):
        result.get_bse_inputs()
