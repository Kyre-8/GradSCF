import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('xc', ['hf', 'svwn'])
def test_native_molecular_response_matches_pyscf_cphf(xc):
    if xc != 'hf':
        pytest.importorskip('jax_xc')
    from gradscf import gto, scf, dft, dfpt
    from pyscf import gto as pgto, scf as pscf, dft as pdft
    from pyscf.scf import cphf
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g')
    mf = (scf.RHF(mol) if xc=='hf' else dft.RKS(mol, xc=xc, grids_level=0)).run()
    ref = mf.to_reference()
    problem = dfpt.DFPT(mf)
    # Identical AO perturbation on independently solved PySCF orbitals.
    b = jnp.array([[.2, .1], [.1, -.1]])
    response = jax.jit(lambda x: problem.response(potential=x))(b)
    assert response.converged
    pmol = pgto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g', verbose=0)
    pmf = pscf.RHF(pmol) if xc=='hf' else pdft.RKS(pmol, xc=xc)
    if xc!='hf': pmf.grids.level=0
    pmf.conv_tol=1e-12
    pmf.kernel()
    c = pmf.mo_coeff; co, cv = c[:,:1], c[:,1:]
    vresp = pmf.gen_response(hermi=1)
    def action(x):
        d = 2 * cv @ x.reshape(1,1) @ co.T
        return cv.T @ vresp(d+d.T) @ co
    x, _ = cphf.solve(action, pmf.mo_energy, pmf.mo_occ, cv.T @ np.asarray(b) @ co, tol=1e-11)
    d = 2 * cv @ x @ co.T
    np.testing.assert_allclose(response.density, d+d.T, atol=2e-7, rtol=1e-6)
