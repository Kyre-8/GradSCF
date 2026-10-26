"""SCF CC consumers share selected blocks, retaining full-frame metadata."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, gto, scf
from gradscf.scf.reference import reference_from_source


@pytest.fixture(scope='module')
def source():
    molecule = gto.M(atom='H 0 0 0; H 0 0 1.4; H 1.5 0 .1; H 1.5 0 1.5',
                     basis='sto-3g', unit='Bohr', cart=True)
    mf = scf.RHF(molecule, integral_backend='native', execution_device='cpu',
        max_memory=128., conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=250).run()
    assert mf.converged
    return mf


def _forbid_full_mo(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('SCF CC consumer requested a complete MO Hamiltonian')
    module = importlib.import_module('gradscf.cc.api')
    monkeypatch.setattr(module, 'reference_from_source', forbidden)
    monkeypatch.setattr(importlib.import_module('gradscf.ao2mo.reference'),
                        'transform_integrals', forbidden)


@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_scf_ground_lambda_density_and_triples_share_snapshot(source, frozen, monkeypatch):
    ref = reference_from_source(source)
    options = dict(frozen=frozen, conv_tol=1e-12, residual_tol=1e-10)
    expected = cc.CCSD(ref, **options).run()
    expected1, expected2 = expected.make_rdm1(), expected.make_rdm2()
    expected_t = expected.ccsd_t()
    _forbid_full_mo(monkeypatch)
    actual = cc.CCSD(source, **options).run()
    assert actual.converged
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-10)
    np.testing.assert_allclose(actual.t1, expected.t1, atol=2e-9)
    np.testing.assert_allclose(actual.t2, expected.t2, atol=2e-9)
    np.testing.assert_allclose(actual.make_rdm1(), expected1, atol=2e-8)
    np.testing.assert_allclose(actual.make_rdm2(), expected2, atol=2e-8)
    np.testing.assert_allclose(actual.ccsd_t(), expected_t, atol=2e-10)
    np.testing.assert_allclose(jnp.trace(actual.make_rdm1()), 4., atol=2e-10)
    assert actual.max_memory == source.max_memory


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_scf_eom_uses_the_ground_snapshot(source, sector, monkeypatch):
    ref = reference_from_source(source)
    expected_ground = cc.CCSD(ref, conv_tol=1e-12, residual_tol=1e-10).run()
    expected = cc.run_eom(ref.h1, ref.eri, expected_ground.result, nocc=2,
                          config=cc.EOMConfig(sector=sector))
    _forbid_full_mo(monkeypatch)
    actual_ground = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
    method = {'ee': actual_ground.EOMEE, 'ip': actual_ground.EOMIP,
              'ea': actual_ground.EOMEA}[sector]
    actual = method().run()
    assert np.all(actual.converged)
    np.testing.assert_allclose(actual.e, expected.energies, atol=2e-8)
    actual.check_source()


@pytest.mark.parametrize('model', ['ccd', 'cc2', 'qcisd'])
def test_other_restricted_models_keep_their_equations(source, model, monkeypatch):
    ref = reference_from_source(source)
    cfg = cc.CCConfig(method=model, conv_tol=1e-12, residual_tol=1e-10)
    expected = cc.run_cc(ref.h1, ref.eri, nocc=2,
        nuclear_repulsion=ref.nuclear_repulsion, config=cfg)
    assert expected.converged
    _forbid_full_mo(monkeypatch)
    actual = cc.CC(source, method=model, conv_tol=1e-12, residual_tol=1e-10).run()
    assert actual.converged
    np.testing.assert_allclose(actual.e_tot, expected.total_energy, atol=3e-10)


def test_failed_scf_rebuild_does_not_leave_a_ready_old_state(source):
    obj = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
    original = source.converged
    try:
        source.converged = False
        with pytest.raises(RuntimeError):
            obj.kernel()
        assert obj.result is None and not obj.converged
        with pytest.raises(RuntimeError, match='ground state'):
            obj.solve_lambda()
    finally:
        source.converged = original


def test_scf_public_energy_alias_does_not_change_the_prepared_hamiltonian(source):
    expected = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
    original = source.e_tot
    try:
        source.e_tot = original+.5
        actual = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
        np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-12)
    finally:
        source.e_tot = original


def test_native_direct_scf_facade_never_materializes_complete_integrals(source, monkeypatch):
    expected = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
    _forbid_full_mo(monkeypatch)
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            pytest.fail('Direct SCF/CC facade requested complete AO ERIs')
        return original(self, operator, *args, **kwargs)
    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    direct = scf.RHF(source.mol, integral_backend='native', execution_device='cpu',
        jk_backend='direct', direct_scf_tol=0., max_memory=128., conv_tol=1e-12,
        conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=250).run()
    assert direct.converged
    actual = cc.CCSD(direct, conv_tol=1e-12, residual_tol=1e-10).run()
    assert actual.converged
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-10)
    density = actual.make_rdm1()
    assert np.isfinite(np.asarray(density)).all()
    np.testing.assert_allclose(jnp.trace(density), 4., atol=2e-10)


@pytest.mark.parametrize('change', ['frozen', 'method', 'coefficients'])
def test_scf_posthoc_consumers_reject_stale_source(source, change):
    obj = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-10).run()
    coefficients = source.mo_coeff
    try:
        if change == 'coefficients':
            source.mo_coeff = coefficients.at[0, 0].add(.01)
        elif change == 'frozen':
            obj.frozen = 1
        else:
            obj.method = 'cc2'
        for operation in (obj.solve_lambda, obj.make_rdm1, obj.make_rdm2, obj.ccsd_t):
            with pytest.raises(RuntimeError, match='kernel'):
                operation()
    finally:
        source.mo_coeff = coefficients
