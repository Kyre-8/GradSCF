"""External JAX Dirac-LDA/PBE, with GradSCF SCF and PySCF response oracle."""
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.dft.derivatives import xc_energy_and_potential_from_density


@pytest.mark.parametrize('xc', ['lda_x', 'pbe'])
def test_external_ks_response_matches_pyscf(xc):
    from gradscf import gto, scf, dfpt
    from gradscf.integrals.molecular.jk import build_j_from_eri_pair_matrix
    from gradscf.scf.uks import run_unrestricted_scf_scan
    from pyscf import gto as pgto, dft as pdft
    from pyscf.scf import cphf
    atom = 'He 0 0 0; H 0 0 1.5'
    mf = scf.RHF(gto.M(atom=atom, charge=1, basis='sto-3g', unit='Bohr'), grids_level=1).run()
    mf.to_reference()
    data = mf._scf_inputs
    args = dict(hcore=data.hcore, overlap=data.overlap, eri=data.eri_pair_matrix,
                ao=data.ao, ao_deriv1=data.ao_deriv1, weights=data.grid_weights, enuc=data.nuclear_repulsion,
                scale=jnp.array(1.))

    def exc(scale, a, d):
        rho = jnp.einsum('gp,spq,gq->sg', a['ao'], d, a['ao'])
        return -.75*(6/jnp.pi)**(1/3)*scale*jnp.sum(a['weights']*jnp.maximum(rho, 1e-18)**(4/3))

    if xc == 'pbe':
        from test_pbe import pbe_xc_energy
        exc = pbe_xc_energy

    def evaluate(d, a):
        potential = xc_energy_and_potential_from_density(a['scale'], molecule=a, density=d, xc_energy_fn=exc)
        total = d.sum(axis=0)
        coulomb = build_j_from_eri_pair_matrix(a['eri'], total)
        e = jnp.sum(a['hcore']*total)+.5*jnp.sum(coulomb*total)+potential.xc_energy+a['enuc']
        f = a['hcore']+coulomb+potential.vxc_matrix
        return e, f

    def builder(d, *_):
        e, f = evaluate(d, args)
        return f, f, e

    occ = jnp.stack([jnp.array([1.,0.])]*2)
    result = run_unrestricted_scf_scan(fock_builder=builder,
        density_spin=jnp.stack([mf.make_rdm1()/2]*2), mo_coeff_spin=jnp.stack([mf.mo_coeff]*2),
        mo_occ_spin=occ, mo_energy_spin=jnp.stack([mf.mo_energy]*2), overlap=data.overlap,
        max_cycle=100, damping=.1, conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, orthogonalization_eps=1e-10)
    density, coeff, _, _, converged, *_ = result
    assert converged
    p = dfpt.ks_problem(coeff, (1,1), args, lambda d,a:evaluate(d,a)[1],
                         lambda d,a:evaluate(d,a)[0], overlap=lambda a:jnp.stack([a['overlap']]*2))
    b = jnp.array([[.2,.1],[.1,-.1]])
    import jax
    tangent = dict(jax.tree.map(jnp.zeros_like, args), hcore=b)
    response = p.response(tangent)
    assert response.converged
    pmol = pgto.M(atom=atom, charge=1, basis='sto-3g', unit='Bohr', verbose=0)
    pmf = pdft.RKS(pmol, xc=xc); pmf.grids.level=1; pmf.conv_tol=1e-12; pmf.kernel()
    np.testing.assert_allclose(evaluate(density, args)[0], pmf.e_tot, atol=2e-7, rtol=0)
    c = pmf.mo_coeff; co, cv = c[:,:1], c[:,1:]
    vresp = pmf.gen_response(hermi=1)
    def action(x):
        d = 2*cv@x.reshape(1,1)@co.T
        return cv.T@vresp(d+d.T)@co
    x, _ = cphf.solve(action, pmf.mo_energy, pmf.mo_occ, cv.T@np.asarray(b)@co, tol=1e-11)
    d = 2*cv@x@co.T
    np.testing.assert_allclose(response.density.sum(axis=0), d+d.T, atol=2e-7, rtol=2e-6)
