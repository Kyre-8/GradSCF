"""The ordinary molecular facades consume external XC energies and parameters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, tdscf
from gradscf.scf.rks import _build_jk


def _model():
    return dft.custom_xc(
        input_fn=lambda state: (state.rho, state.tau, state.weights),
        energy_fn=lambda params, inputs: params['scale']*jnp.dot(
            inputs[2], .08*inputs[0]**2+.015*inputs[1]))


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_external_functional_facade_solves_actual_energy_fock_and_td(spin):
    restricted = spin == 'restricted'
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',
                charge=0 if restricted else 1, spin=0 if restricted else 1)
    model, params = _model(), {'scale': jnp.asarray(.3)}
    cls = dft.RKS if restricted else dft.UKS
    mf = cls(mol, xc=model, params=params, grids_level=0,
             max_cycle=60, conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-8).run()
    assert mf.converged
    reference = mf.to_reference()
    density_spin = jnp.asarray(reference.rdm1)
    density = density_spin.sum(axis=0)
    eri = reference.eri_pair_matrix
    if eri is None:
        eri = reference.rep_tensor
    coulomb = _build_jk(eri, density)[0]
    xc = model.energy_from_molecule(params, reference)
    expected_energy = (jnp.einsum('ij,ij->', density, reference.h1e)
        + .5*jnp.einsum('ij,ij->', density, coulomb)+xc+reference.nuclear_repulsion)
    potential = model.potential(params, reference, density_spin)
    expected_fock = reference.h1e+coulomb+(potential.mean(axis=0) if restricted else potential)
    np.testing.assert_allclose(mf.e_tot, expected_energy, rtol=0, atol=2e-10)
    np.testing.assert_allclose(reference.fock_matrix, expected_fock, rtol=0, atol=2e-10)
    assert np.isfinite(reference.fock_matrix).all()
    assert abs(float(xc)) > 1e-5
    automatic_solver = tdscf.TDA(mf, nstates=1)
    automatic = automatic_solver.kernel().excitation_energies
    explicit = tdscf.TDA(reference, xc_functional=model, xc_params=params,
                        nstates=1).kernel().excitation_energies
    np.testing.assert_allclose(automatic, explicit, rtol=0, atol=2e-10)

    def excitation(scale):
        return tdscf.TDA(reference, xc_functional=model, xc_params={'scale': scale},
                         nstates=1).kernel().excitation_energies[0]

    actual = jax.grad(excitation)(params['scale'])
    step = 1e-4
    expected = (excitation(.3+step)-excitation(.3-step))/(2*step)
    assert abs(float(actual)) > 1e-6
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-9)
    mf.params = {'scale': jnp.asarray(.31)}
    with pytest.raises(RuntimeError, match='configuration changed'):
        tdscf.TDA(mf).molecule
    mf.run()
    reused = automatic_solver.kernel().excitation_energies
    current = tdscf.TDA(mf, nstates=1).kernel().excitation_energies
    np.testing.assert_allclose(reused, current, rtol=0, atol=2e-10)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_traditional_functional_object_matches_named_facade(spin):
    restricted = spin == 'restricted'
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',
                charge=0 if restricted else 1, spin=0 if restricted else 1)
    cls = dft.RKS if restricted else dft.UKS
    options = dict(grids_level=0, conv_tol=1e-11, conv_tol_grad=1e-8)
    actual = cls(mol, xc=dft.custom_xc('pbe'), **options).run()
    expected = cls(mol, xc='pbe', **options).run()
    assert actual.converged and expected.converged
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, rtol=0, atol=2e-10)
    np.testing.assert_allclose(actual.make_rdm1(), expected.make_rdm1(), rtol=0, atol=2e-9)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_custom_laplacian_input_is_available_from_current_density(spin):
    restricted = spin == 'restricted'
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',
                charge=0 if restricted else 1, spin=0 if restricted else 1)
    model = dft.Functional(
        input_fn=lambda state: (state.laplacian_rho, state.weights),
        energy_fn=lambda params, inputs: params['scale']*jnp.dot(inputs[1], inputs[0]**2))
    params = {'scale': jnp.asarray(.002)}
    mf = (dft.RKS if restricted else dft.UKS)(mol, xc=model, params=params,
        max_cycle=40, conv_tol=1e-11, conv_tol_grad=1e-8).run()
    assert mf.converged
    reference = mf.to_reference()
    assert reference.ao_laplacian is not None
    assert np.isfinite(reference.ao_laplacian).all()
    value = model.inputs_for_density(reference, reference.rdm1)[0]
    doubled = model.inputs_for_density(reference, 2*reference.rdm1)[0]
    np.testing.assert_allclose(doubled, 2*value, rtol=0, atol=2e-12)
    assert np.isfinite(model.potential(params, reference, reference.rdm1)).all()


def test_custom_finite_energy_with_nonfinite_potential_is_not_converged():
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g')
    # sqrt(0) has a finite scalar value and an undefined derivative. This
    # deliberately invalid callback must remain visible at the facade.
    model = dft.Functional(input_fn=lambda state: state.rho,
        energy_fn=lambda params, rho: jnp.sqrt(jnp.sum(rho*0.)))
    mf = dft.RKS(mol, xc=model, params={}, max_cycle=8)
    try:
        mf.run()
    except FloatingPointError:
        assert not mf.converged
        return
    assert not mf.converged
    assert not np.isfinite(mf.scf_result.fock_matrix).all()
    with pytest.raises(RuntimeError, match='converge'):
        mf.to_reference()
