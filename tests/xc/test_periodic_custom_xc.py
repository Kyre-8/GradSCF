"""One periodic XC port retains complete recipes and external callbacks."""
import jax.numpy as jnp
import numpy as np
import pytest


def test_periodic_traditional_object_keeps_full_rsh_operator():
    from gradscf.dft.xc import TraditionalXCFunctional
    from gradscf.pbc import gto,dft,_kpoint
    from gradscf.integrals.periodic.fft import build_inputs
    spec='RSH(.4,.8,-.6)'
    model=TraditionalXCFunctional(spec)
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    points=cell.make_kpts((2,1,1));inputs=build_inputs(cell,kpts=points)
    density=jnp.broadcast_to(jnp.array([[.3,.04],[.04,.2]]),(2,2,2,2)).astype(complex)
    evaluate=lambda xc:_kpoint.energy_and_fock(density,inputs,mesh=cell.mesh,xc=xc)
    for actual,expected in zip(evaluate(model),evaluate(spec)):
        np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
    controls=dict(conv_tol=1e-11,conv_tol_density=1e-10,conv_tol_grad=1e-9,max_cycle=100,damping=.1)
    reference=dft.KRKS(cell,kpts=points,xc=spec).run(**controls)
    actual=dft.KRKS(cell,kpts=points,xc=model).run(**controls)
    assert actual.converged and reference.converged
    np.testing.assert_allclose(actual.e_tot,reference.e_tot,atol=2e-13,rtol=0)
    np.testing.assert_allclose(actual.make_rdm1(),reference.make_rdm1(),atol=2e-12,rtol=0)


def test_periodic_external_callback_feature_forms_still_agree():
    from gradscf.dft import Functional
    from gradscf.pbc import gto,dft
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    models=(
        Functional(lambda x:jnp.stack([x.rho,x.weights]),
                   lambda p,x:p*jnp.sum(x[1]*x[0]**2)),
        Functional(lambda x:{'rho':x.rho,'weights':x.weights},
                   lambda p,x:p*jnp.sum(x['weights']*x['rho']**2)),
        Functional(lambda x:x,lambda p,x:p*jnp.sum(x.weights*x.rho**2)),
    )
    results=[dft.RKS(cell,functional=model,params=.03).run(
        conv_tol=1e-11,conv_tol_density=1e-10,conv_tol_grad=1e-9,max_cycle=100,damping=.1)
        for model in models]
    assert all(result.converged for result in results)
    for result in results[1:]:
        np.testing.assert_allclose(result.e_tot,results[0].e_tot,atol=2e-13,rtol=0)
        np.testing.assert_allclose(result.make_rdm1(),results[0].make_rdm1(),atol=2e-12,rtol=0)
    with pytest.raises(ValueError,match='no named xc baseline'):
        dft.RKS(cell,xc='pbe',functional=models[0],params=.03)


def test_periodic_xc_callback_alias_retains_params_and_rejects_duplicates():
    from gradscf.dft import Functional
    from gradscf.pbc import gto,dft
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    model=Functional(lambda x:x,lambda p,x:p*jnp.sum(x.weights*x.rho**2))
    actual=dft.RKS(cell,xc=model,params=.03)
    reference=dft.RKS(cell,functional=model,params=.03)
    for result in (actual,reference):
        result.run(conv_tol=1e-11,conv_tol_density=1e-10,conv_tol_grad=1e-9,max_cycle=100,damping=.1)
        assert result.converged
    np.testing.assert_allclose(actual.e_tot,reference.e_tot,atol=2e-13,rtol=0)
    np.testing.assert_allclose(actual.make_rdm1(),reference.make_rdm1(),atol=2e-12,rtol=0)
    with pytest.raises(ValueError,match='(one|both|duplicate|Specify)'):
        dft.RKS(cell,xc=model,functional=model,params=.03)


@pytest.mark.parametrize('consumer',['energy','scf','q_kernel'])
def test_periodic_array_callbacks_reject_nonhf_named_baseline(consumer):
    from gradscf.dft import Functional
    from gradscf.pbc import gto,_kpoint
    from gradscf.pbc.functional import density_inputs
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.response import q_kernel
    from gradscf.integrals.periodic.fft import build_inputs
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    inputs=build_inputs(cell)
    energy=lambda p,x:p*jnp.dot(x.weights,x.rho**2)
    q_energy=lambda p,a,d,t,z:energy(p,density_inputs(a,d+z+z.conj().swapaxes(-1,-2)))
    model=Functional(lambda x:x,energy,q_energy_fn=q_energy)
    density=jnp.broadcast_to(jnp.eye(2)*.3,(2,1,2,2)).astype(complex)
    options=dict(mesh=cell.mesh,xc='pbe',functional=model,params=.03)
    with pytest.raises(ValueError,match='(baseline|complete XC)'):
        if consumer=='energy':_kpoint.energy_and_fock(density,inputs,**options)
        elif consumer=='scf':_kpoint.run_kpoint_scf(inputs,nelec=cell.nelec,max_cycle=1,**options)
        else:q_kernel(inputs,density,momentum_transfer(np.zeros((1,3)),np.zeros(3)),**options)


def test_periodic_array_callback_hf_default_is_only_a_placeholder():
    from gradscf.dft import Functional
    from gradscf.pbc import gto,_kpoint
    from gradscf.integrals.periodic.fft import build_inputs
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    inputs=build_inputs(cell)
    model=Functional(lambda x:x,lambda p,x:p*jnp.dot(x.weights,x.rho**2))
    density=jnp.broadcast_to(jnp.eye(2)*.3,(2,1,2,2)).astype(complex)
    options=dict(mesh=cell.mesh,functional=model,params=.03)
    for actual,expected in zip(_kpoint.energy_and_fock(density,inputs,**options),
                               _kpoint.energy_and_fock(density,inputs,xc=None,**options)):
        np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
