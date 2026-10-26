"""Weighted GGA contraction with an independent handwritten PBE test oracle."""
import jax
import jax.numpy as jnp
import numpy as np
from test_pbe import pbe_energy_density


def test_periodic_pbe_weighted_energy_derivative(monkeypatch):
    from gradscf.pbc import gto,_kpoint
    from gradscf.integrals.periodic.fft import build_inputs
    def point_kernel(xc,kind):
        assert xc=='pbe' and kind=='GGA'
        return jax.vmap(jax.value_and_grad(lambda v:pbe_energy_density(v[:2],v[2:].reshape(2,3))))
    monkeypatch.setattr(_kpoint,'_point_unrestricted_xc_value_and_grad_kernel',point_kernel)
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    data=build_inputs(cell,kpts=cell.make_kpts((2,1,1)),kweights=[.3,.7])
    d=jnp.broadcast_to(jnp.eye(2)*.2,(2,2,2,2))
    b=jnp.broadcast_to(jnp.array([[.2,.1],[.1,-.1]]),(2,2,2,2))
    e,f=_kpoint.energy_and_fock(d,data,mesh=cell.mesh,xc='pbe')
    derivative=jax.grad(lambda t:_kpoint.energy_and_fock(d+t*b,data,mesh=cell.mesh,xc='pbe')[0])(0.)
    expected=jnp.einsum('k,skij,skji->',data.kweights,b,f).real
    np.testing.assert_allclose(derivative,expected,atol=1e-10)
    from gradscf.scf import SCFDifferentiationConfig
    def run(t):
        return _kpoint.run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,xc='pbe',
            temperature=t,exxdiv=None,max_cycle=120,damping=.2,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,
            differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-10,max_iter=80))
    ground=run(.1)
    assert ground.converged
    derivative=jax.jit(jax.grad(lambda t:run(t).free_energy))(.1)
    np.testing.assert_allclose(derivative,-ground.entropy,atol=1e-7)
