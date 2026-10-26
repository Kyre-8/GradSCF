"""Geometry-driven response includes the moving metric and Pulay forces."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import dfpt,ofdft
from gradscf.pbc import gto,scf


@pytest.mark.parametrize('model',['hf','thermal','ofdft','ofdft_gaussian','pbe'])
def test_atomic_response_and_force_constants_match_displaced_scf(model):
    if model=='pbe':pytest.importorskip('jax_xc',reason='Production PBE geometry response requires jax-xc')
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
    def calculation(c):
        if model.startswith('ofdft'):
            return ofdft.OFDFT(c,xc=None,representation='periodic_gaussian' if model.endswith('gaussian') else 'periodic',
                               tolerance=1e-9,maxiter=500).run()
        from gradscf.pbc import dft
        cls=dft.RKS if model=='pbe' else scf.RHF
        return cls(c,temperature=.1 if model=='thermal' else None).run(
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    mf=calculation(cell)
    problem=dfpt.atomic_problem(mf)
    assert problem._base()[1]
    values=problem.quantities(problem.solution,problem.inputs)
    gradient=mf.nuc_grad_method();gradient.kernel()
    np.testing.assert_allclose(values['forces'],gradient.result.forces,atol=2e-7)
    direction=jnp.zeros_like(cell.coords).at[1,0].set(1.)
    response=problem.response(tangent=direction)
    assert response.converged
    def shifted(t):
        displaced=gto.M(atom=[('H',r) for r in np.asarray(cell.coords+t*direction)],
                         a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
        other=calculation(displaced)
        driver=other.nuc_grad_method();driver.kernel()
        density=other.result.density if model.startswith('ofdft') else other.result.density_spin
        return density,driver.result.forces
    plus,minus=shifted(1e-4),shifted(-1e-4)
    np.testing.assert_allclose(response.density,(plus[0]-minus[0])/2e-4,atol=3e-6)
    np.testing.assert_allclose(response.values['forces'],(plus[1]-minus[1])/2e-4,atol=5e-6)
