import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('nk', [1, 2])
@pytest.mark.parametrize('xc', ['hf', 'svwn'])
def test_periodic_ks_q0_matches_perturbed_scf(nk, xc):
    if xc != 'hf':
        pytest.importorskip('jax_xc')
    from gradscf import dfpt
    from gradscf.pbc import gto, dft, scf
    from gradscf.pbc.scf import run_gamma_scf
    from gradscf.pbc._kpoint import run_kpoint_scf
    cell = gto.M(atom='H .2 .3 .4; H 1.6 .3 .4', a=np.eye(3)*6, unit='Bohr',
                 basis='gth-szv', pseudo='gth-pade', mesh=(15,)*3)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150)
    cls = (scf.RHF if nk==1 else scf.KRHF) if xc=='hf' else (dft.RKS if nk==1 else dft.KRKS)
    mf = cls(cell, kpts=cell.make_kpts((nk,1,1)), xc=xc).run(**controls)
    assert mf.converged
    problem = dfpt.DFPT(mf)
    perturbation = jnp.tile(jnp.array([[.2, .1], [.1, -.1]])[None], (nk,1,1))
    result = problem.response(potential=perturbation)
    assert result.converged
    def solve(t):
        data = mf.inputs._replace(hcore=mf.inputs.hcore+t*perturbation)
        if nk==1:
            return run_gamma_scf(data, nelec=cell.nelec, xc=xc, **controls)
        return run_kpoint_scf(data, mesh=cell.mesh, nelec=cell.nelec, xc=xc, **controls)
    plus, minus = solve(1e-4), solve(-1e-4)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(result.density, (plus.density_spin-minus.density_spin)/2e-4, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(result.values['electron_number'], 0., atol=1e-9)
    if nk == 1 and xc == 'hf':
        series = problem.series(potential=perturbation, order=2)
        assert series.converged
        np.testing.assert_allclose(series.density[0], result.density, atol=1e-8)
        np.testing.assert_allclose(2*series.energy[1],
            jnp.einsum('skij,kji->', result.density, perturbation).real/nk, atol=1e-8)
    if nk==1:
        with pytest.raises(ValueError, match='not closed'):
            dfpt.DFPT(mf, q=[.5, 0., 0.])
    else:
        assert dfpt.DFPT(mf,q=[.5,0.,0.]).response(potential=perturbation).converged
