"""External PBE uses the public Functional/periodic KS path without jax-xc."""
import jax.numpy as jnp
import numpy as np
import pytest
from test_pbe import pbe_energy_density


@pytest.mark.parametrize('nk,temperature',[(1,None),(2,.12)])
def test_public_periodic_pbe_matches_pyscf(nk,temperature):
    pytest.importorskip('pyscf')
    from gradscf import dft
    from gradscf.pbc import gto,dft as pdft
    from pyscf.pbc import gto as pgto,dft as reference_dft
    from pyscf.pbc.scf import addons
    model=dft.Functional(lambda s:(s.rho_spin,s.grad_rho_spin,s.weights),
        lambda p,x:jnp.sum(x[2]*pbe_energy_density(x[0],jnp.moveaxis(x[1],-1,1))))
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(31,)*3,
            basis='gth-szv',pseudo='gth-pade',precision=1e-10)
    cell=gto.M(**kw);points=cell.make_kpts((nk,1,1))
    ours=(pdft.RKS if nk==1 else pdft.KRKS)(cell,kpts=points,functional=model,
        temperature=temperature).run(max_cycle=120,damping=.1,conv_tol=1e-12,
                                    conv_tol_density=1e-10,conv_tol_grad=1e-9)
    refcell=pgto.M(**kw,cart=True,verbose=0)
    ref=reference_dft.RKS(refcell) if nk==1 else reference_dft.KRKS(refcell,kpts=np.asarray(points))
    ref.xc='pbe';ref.conv_tol=1e-12;ref.conv_tol_grad=1e-9
    if temperature is not None:ref=addons.smearing_(ref,sigma=temperature,method='fermi')
    ref.kernel()
    assert ours.converged and ref.converged
    np.testing.assert_allclose(ours.e_tot,ref.e_tot,atol=1e-6,rtol=0)
    if temperature is not None:
        np.testing.assert_allclose(ours.e_free,ref.e_free,atol=1e-6,rtol=0)
