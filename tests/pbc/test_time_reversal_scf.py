import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,scf


def test_time_reversal_checks_self_paired_points():
    from gradscf.pbc.kpoints import time_reversal_plan
    plan=time_reversal_plan(np.array([[0.,0.,0.],[.5,0.,0.]]))
    real=jnp.broadcast_to(jnp.eye(2,dtype=complex),(2,2,2))
    assert plan.consistent(real)
    magnetic=real.at[0,0,1].set(.1j).at[0,1,0].set(-.1j)
    assert not plan.consistent(magnetic)


def test_time_reversal_rejects_an_incompatible_external_potential():
    from gradscf import dft
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.pbc.kpoints import time_reversal_plan
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    data=build_inputs(cell)
    model=dft.Functional(lambda s:s.density_matrix,lambda p,d:.1*jnp.imag(d[0,0,0,1]))
    result=scf.run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,functional=model,
        reduction=time_reversal_plan(np.zeros((1,3))),max_cycle=3,exxdiv=None)
    assert not result.converged


@pytest.mark.parametrize('temperature',[None,.1])
def test_time_reversal_scf_matches_full_mesh(temperature):
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(11,)*3)
    points=cell.make_kpts((4,1,1),shift=(.5,0,0))
    controls=dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=120,damping=.1)
    full=scf.KRHF(cell,kpts=points,temperature=temperature,exxdiv=None).run(**controls)
    reduced=scf.KRHF(cell,kpts=points,temperature=temperature,exxdiv=None,time_reversal=True).run(**controls)
    assert full.converged and reduced.converged
    assert len(reduced.reduction.representatives)==2
    np.testing.assert_allclose(reduced.e_free,full.e_free,atol=1e-10)
    np.testing.assert_allclose(reduced.make_rdm1(),full.make_rdm1(),atol=1e-9)


def test_time_reversal_implicit_response_uses_full_state():
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.pbc.kpoints import time_reversal_plan
    from gradscf.scf import SCFDifferentiationConfig
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    points=cell.make_kpts((2,1,1),shift=(.5,0,0))
    data=build_inputs(cell,kpts=points)
    plan=time_reversal_plan(np.asarray(points)@np.asarray(cell.lattice).T/(2*np.pi))
    b=jnp.array([[[.1,.03j],[-.03j,-.1]],[[.2,.02j],[-.02j,-.2]]])
    def solve(t,reduction):
        return scf.run_kpoint_scf(data._replace(hcore=data.hcore+t*b),mesh=cell.mesh,nelec=cell.nelec,
            temperature=.1,exxdiv=None,reduction=reduction,max_cycle=120,damping=.1,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,
            differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-11,max_iter=80))
    # This perturbation breaks TR. Implicit backward must use the full residual,
    # even when the unperturbed primal used representative-point diagonalization.
    assert solve(0.,plan).converged
    slope=jax.grad(lambda t:jnp.real(solve(t,plan).density_spin[0,0,0,0]))(0.)
    fd=jnp.real((solve(1e-5,None).density_spin[0,0,0,0]-solve(-1e-5,None).density_spin[0,0,0,0])/2e-5)
    assert abs(slope)>1e-5
    np.testing.assert_allclose(slope,fd,atol=1e-7,rtol=1e-5)
