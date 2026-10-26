"""Retarded optical tensors, units, and response through BSE poles/residues."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import bse
from test_full import model


def system(t=0.):
    qp,e,l,d=model()
    space=bse.make_bse_space(5,2)
    result=bse.run_bse(qp*(1+.1*t),e,l*(1+.05*t),space,
        config=bse.BSEConfig(nroots=2,tda=False,solver='dense'))
    return result,d*(1+.03*t),space


def test_polarizability_has_static_and_causal_limits():
    result,d,space=system()
    mu=np.asarray(bse.transition_dipoles(result,d,space))
    energies=np.asarray(result.excitation_energies)
    static=bse.polarizability(result,d,space,0.)
    np.testing.assert_allclose(static,2*np.einsum('si,sj,s->ij',mu,mu,1/energies),atol=1e-12)
    w=jnp.array([.2,.4,.6])
    tensor=bse.polarizability(result,d,space,w,eta=.02)
    np.testing.assert_allclose(bse.polarizability(result,d,space,-w,eta=.02),tensor.conj(),atol=1e-12)
    reference=np.zeros((3,3,3),complex)
    for k,f in enumerate(w):
        for energy,dipole in zip(energies,mu):
            reference[k]+=np.outer(dipole,dipole)*(1/(energy-f-.02j)+1/(energy+f+.02j))
    np.testing.assert_allclose(tensor,reference,atol=1e-12)
    assert np.linalg.eigvalsh(np.imag(tensor)).min()>-1e-12


def test_cross_section_and_polarization():
    result,d,space=system()
    w=jnp.array([.2,.4,.6])
    alpha=bse.polarizability(result,d,space,w,eta=.02)
    cross=bse.absorption_cross_section(result,d,space,w,eta=.02)
    expected=4*np.pi*w/137.035999084*np.trace(np.imag(alpha),axis1=-2,axis2=-1)/3
    np.testing.assert_allclose(cross,expected,atol=1e-12)
    directional=[bse.absorption_cross_section(result,d,space,w,eta=.02,polarization=v) for v in np.eye(3)]
    np.testing.assert_allclose(cross,np.mean(directional,axis=0),atol=1e-12)
    np.testing.assert_allclose(bse.absorption_cross_section(result,d,space,w,eta=.02,unit='Mb'),cross*28.0028520539,rtol=1e-10)
    assert np.isnan(bse.absorption_cross_section(result,d,space,w,polarization=jnp.zeros(3))).all()


def test_optical_spectrum_parameter_jvp_vjp():
    def loss(t):
        result,d,space=system(t)
        return jnp.sum(bse.absorption_cross_section(result,d,space,jnp.array([.2,.5]),eta=.03))
    value,grad=jax.jit(jax.value_and_grad(loss))(0.)
    assert np.isfinite(value)
    np.testing.assert_allclose(jax.jvp(loss,(0.,),(1.,))[1],grad,atol=1e-9)
    np.testing.assert_allclose(grad,(loss(1e-4)-loss(-1e-4))/2e-4,atol=1e-8)
