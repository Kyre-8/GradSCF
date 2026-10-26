"""Prevent newly recognized recipes from bypassing method support checks."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('method', ['g0w0', 'evgw'])
def test_actual_rsh_uks_reference_reaches_molecular_gw_and_bse(method):
    from gradscf import bse, dft, gto, gw
    from gradscf.scf.reference import as_reference
    mf = dft.UKS(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g'),
        xc='cam-b3lyp', grids_level=0, conv_tol=1e-11).run()
    assert mf.converged
    with pytest.raises(NotImplementedError, match='range|Range'):
        as_reference(mf)
    assert as_reference(mf, allow_range_separated=True) is mf.reference
    calculation = gw.UGW(mf, method=method, nw=16, max_cycle=60,
                          conv_tol=1e-7, damp=.3).run()
    assert calculation.converged
    response = bse.BSE(calculation, nroots=1).run()
    assert np.asarray(response.converged).all()
    assert np.isfinite(response.e).all()


def test_ofdft_rejects_a_recipe_whose_dispersion_is_not_implemented():
    from gradscf import ofdft
    data = ofdft.periodic_inputs(jnp.eye(3)*5., (3, 3, 3), nelectron=2.)
    with pytest.raises(NotImplementedError, match='dispersion'):
        ofdft.run_ofdft(data, config=ofdft.OFDFTConfig(xc='b97d', maxiter=1))


@pytest.mark.parametrize('consumer', ['query', 'gamma', 'optics', 'export'])
def test_deferred_periodic_gw_and_optics_reject_new_screened_sources(consumer, monkeypatch, tmp_path):
    from gradscf.pbc import scf
    monkeypatch.setattr(scf, 'validate_reference', lambda *args, **kwargs: None)
    source = SimpleNamespace(xc='hse06', exxdiv='ewald', unrestricted=False,
        cell=SimpleNamespace(dimension=3), inputs=SimpleNamespace(boundary=None))
    with pytest.raises(NotImplementedError, match='range|screened|LDA/GGA|hybrid'):
        if consumer == 'query':
            from gradscf.gw.pbc.query import _source
            _source(source, False)
        elif consumer == 'gamma':
            from gradscf.gw.pbc.krgw import _validate_gw_reference
            _validate_gw_reference(source)
        elif consumer == 'optics':
            from gradscf.bse.pbc.properties import ks_dipoles
            ks_dipoles(source)
        else:
            from gradscf_tools.berkeleygw import export_scf
            export_scf(source, tmp_path)


def test_restricted_stability_rejects_missing_screened_curvature():
    from gradscf import dft, gto
    from gradscf.scf.stability import restricted_stability
    source = dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g'),
                     xc='cam-b3lyp', grids_level=0, conv_tol=1e-11).run()
    assert source.converged
    with pytest.raises(NotImplementedError, match='range|screened'):
        restricted_stability(source)


def _custom_source(spin):
    from gradscf import dft, gto
    functional = dft.custom_xc(
        input_fn=lambda state: (state.rho, state.weights),
        energy_fn=lambda params, inputs: params['a']*jnp.dot(inputs[1], inputs[0]**2))
    cls = dft.RKS if spin == 'restricted' else dft.UKS
    source = cls(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g'),
                 xc=functional, params={'a': jnp.asarray(.02)},
                 grids_level=0, conv_tol=1e-11).run()
    assert source.converged
    return source


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_custom_stability_does_not_use_the_hf_initialization_placeholder(spin):
    source = _custom_source(spin)
    with pytest.raises(NotImplementedError, match='external|custom'):
        source.stability()


def test_custom_molecular_dfpt_requires_its_full_energy_adapter():
    from gradscf import dfpt
    source = _custom_source('restricted')
    with pytest.raises(NotImplementedError, match='external|Functional|functional'):
        dfpt.DFPT(source)


def test_noncollinear_xc_preserves_invalid_local_energy_and_potential(monkeypatch):
    from gradscf.scf import gks

    def factory(*args):
        return lambda fields: (jnp.full(fields.shape[0], jnp.nan),
                               jnp.full(fields.shape, jnp.nan))

    monkeypatch.setattr(gks, 'grid_value_and_grad', factory)
    config = gks.GKSConfig(xc_spec='lda', collinear='ncol')
    energy, potential = jax.jit(lambda density: gks._generalized_xc(
        density, jnp.ones((2, 1)), jnp.zeros((4, 2, 1)),
        jnp.ones(2), config, 'LDA'))(jnp.eye(2)*.3)
    assert not np.isfinite(energy)
    assert not np.isfinite(potential).all()


def test_unrestricted_energy_only_scan_does_not_certify_nonfinite_orbitals():
    from gradscf.scf.uks import run_unrestricted_scf_scan
    density = jnp.tile(jnp.diag(jnp.asarray([1., 0.])), (2, 1, 1))
    coeff = jnp.tile(jnp.eye(2), (2, 1, 1))
    orbital_energy = jnp.tile(jnp.asarray([-1., 1.]), (2, 1))
    def builder(*args):
        fock = jnp.full_like(density, jnp.nan)
        return fock, fock, jnp.asarray(0.)
    result = run_unrestricted_scf_scan(fock_builder=builder,
        density_spin=density, mo_coeff_spin=coeff,
        mo_occ_spin=jnp.asarray([[1., 0.], [1., 0.]]),
        mo_energy_spin=orbital_energy, overlap=jnp.eye(2),
        max_cycle=3, damping=0., conv_tol=1e-10, conv_tol_density=1e-10,
        orthogonalization_eps=1e-12, convergence_metric='energy')
    assert not bool(result[4])
    assert not np.isfinite(result[1]).all()
