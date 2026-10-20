"""Screened operators retain the existing complex finite-q block convention."""
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('exxdiv',[None,'ewald'])
def test_nonzero_q_screened_exchange_decomposes_and_respects_adjoint(exxdiv):
    from gradscf.pbc import gto
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.response import q_kernel
    from gradscf.integrals.periodic.fft import build_inputs
    # Full/SR exchange needs its high-G tail converged for q/-q adjointness:
    # this fixed cell gives 1.4e-6 at 7^3 and 6.5e-14 at 31^3.
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(31,)*3)
    points=cell.make_kpts((3,1,1));inputs=build_inputs(cell,kpts=points)
    fractional=np.asarray(points)@np.asarray(cell.lattice).T/(2*np.pi)
    transfer=momentum_transfer(fractional,[1/3,0.,0.])
    negative=momentum_transfer(fractional,[-1/3,0.,0.])
    density=jnp.broadcast_to(jnp.eye(2)*.3,(2,3,2,2)).astype(complex)
    random=np.random.default_rng(971)
    tangent=jnp.asarray(random.normal(size=density.shape)+1j*random.normal(size=density.shape))*.03
    apply=lambda spec,t=transfer:q_kernel(inputs,density,t,mesh=cell.mesh,xc=spec,exxdiv=exxdiv)
    full=apply('hf')(tangent);hartree=apply('0*hf')(tangent)
    short=apply('SR_HF(.4)')(tangent);long=apply('LR_HF(.4)')(tangent)
    np.testing.assert_allclose(short+long-full,hartree,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(apply('LR_HF(.4)')(1j*tangent),1j*long,atol=2e-13,rtol=2e-13)
    inverse=np.argsort(transfer.partners)
    adjoint=lambda value:value[:,inverse].conj().swapaxes(-1,-2)
    np.testing.assert_allclose(apply('SR_HF(.4)',negative)(adjoint(tangent)),
                               adjoint(short),atol=2e-12,rtol=2e-12)
