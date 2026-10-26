"""Aligned-grid molecular MGGA SCF and complete A/B reference checks."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import RKSConfig, UKSConfig, run_rks_from_integrals, run_uks_from_integrals
from gradscf import tdscf


CUSTOM_RSH = '0.2*HF + RSH(0.3,0.6,-0.2) + 0.7*pbe'


@pytest.fixture(scope='module', params=[
    ('restricted', 'r2scan'), ('unrestricted', 'r2scan'),
    ('restricted', 'hse06'), ('unrestricted', 'hse06'),
    ('restricted', 'lc-wpbe'), ('unrestricted', 'lc-wpbe'),
    ('restricted', CUSTOM_RSH), ('unrestricted', CUSTOM_RSH)],
    ids=lambda item: item[0]+'-custom-rsh' if 'RSH(' in item[1] else '-'.join(item))
def aligned_reference(request):
    from dataclasses import replace
    from pyscf import dft, gto
    from pyscf_reference import restricted_reference_from_pyscf, unrestricted_reference_from_pyscf

    spin, xc = request.param
    restricted = spin == 'restricted'
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g',
                charge=0 if restricted else 1, spin=0 if restricted else 1, cart=True, verbose=0)
    mf = (dft.RKS if restricted else dft.UKS)(mol)
    mf.xc, mf.grids.level = xc.replace('lc-wpbe', 'lc_wpbe'), 0
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    assert mf.converged
    reference = (restricted_reference_from_pyscf if restricted else unrestricted_reference_from_pyscf)(mf)
    reference = replace(reference, xc_spec=xc)
    from gradscf.scf.rsh import rsh_preset
    preset = rsh_preset(xc)
    if preset is not None:
        with mol.with_range_coulomb(preset.default_omega):
            eri_lr = jnp.asarray(mol.intor('int2e'))
        reference = replace(reference, eri_lr=eri_lr, xc_spec=xc,
            range_separated_coefficients=preset.to_range_separated_hybrid_coefficients())
    return restricted, xc, mf, reference


def test_aligned_standard_scf_energy_density_and_fock_match_pyscf(aligned_reference):
    restricted, xc, mf, reference = aligned_reference
    common = dict(overlap=mf.get_ovlp(), hcore=mf.get_hcore(), eri=reference.rep_tensor,
        nuclear_repulsion=mf.mol.energy_nuc(), ao=reference.ao,
        ao_deriv1=reference.ao_deriv1, grid_weights=reference.grid.weights,
        eri_lr=reference.eri_lr)
    if restricted:
        result = run_rks_from_integrals(**common, nelectron=mf.mol.nelectron,
            init_density=mf.make_rdm1(), config=RKSConfig(xc_spec=xc,
                conv_tol=1e-11, conv_tol_grad=1e-8, max_cycle=20))
        density, fock = result.density_matrix, result.fock_matrix
    else:
        result = run_uks_from_integrals(**common, nalpha=mf.mol.nelec[0], nbeta=mf.mol.nelec[1],
            init_density_alpha=mf.make_rdm1()[0], init_density_beta=mf.make_rdm1()[1],
            config=UKSConfig(xc_spec=xc, conv_tol=1e-11, conv_tol_grad=1e-8, max_cycle=20))
        density = jnp.stack([result.density_matrix_alpha, result.density_matrix_beta])
        fock = jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
    assert result.converged
    np.testing.assert_allclose(result.total_energy, mf.e_tot, rtol=0, atol=2e-8)
    np.testing.assert_allclose(density, mf.make_rdm1(), rtol=0, atol=2e-7)
    np.testing.assert_allclose(fock, mf.get_fock(), rtol=0, atol=2e-7)


def _spin_block_matrix(blocks):
    aa, ab, bb = map(np.asarray, blocks)
    na, nb = int(np.prod(aa.shape[:2])), int(np.prod(bb.shape[:2]))
    return np.block([[aa.reshape(na, na), ab.reshape(na, nb)],
                     [ab.transpose(2, 3, 0, 1).reshape(nb, na), bb.reshape(nb, nb)]])


def test_complete_standard_td_a_b_match_pyscf(aligned_reference):
    restricted, xc, mf, reference = aligned_reference
    expected_a, expected_b = mf.TDDFT().get_ab()
    if restricted:
        dimension = int(np.prod(expected_a.shape[:2]))
        expected_a, expected_b = expected_a.reshape(dimension, dimension), expected_b.reshape(dimension, dimension)
    else:
        expected_a, expected_b = _spin_block_matrix(expected_a), _spin_block_matrix(expected_b)
        dimension = expected_a.shape[0]
    operator = tdscf.TDDFT(reference)._build_solver().gen_tdhf_vind()
    identity = jnp.eye(dimension)
    zeros = jnp.zeros_like(identity)
    actual_a = np.asarray(operator(jnp.concatenate([identity, zeros], axis=-1)))[:, :dimension].T
    actual_b = np.asarray(operator(jnp.concatenate([zeros, identity], axis=-1)))[:, :dimension].T
    np.testing.assert_allclose(actual_a, expected_a, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(actual_b, expected_b, rtol=2e-7, atol=2e-8)
