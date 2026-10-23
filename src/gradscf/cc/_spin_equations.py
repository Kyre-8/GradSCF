# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE.pyscf).
# Adapted from PySCF 2.9.0; see NOTICE.md for attribution and changes.
"""Physical spin-orbital CCSD residual; iteration lives in solvers."""
import jax.numpy as np
from jax.numpy import einsum
from . import _spin_intermediates as imd

def residual(t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)


    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*imd.contract_Wvvvv(t1, t2, eris, tau, tau=tau)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    return t1new, t2new


def correlation_energy(t1, t2, eris):
    no = t1.shape[0]
    return (einsum('ia,ia', eris.fock[:no, no:], t1)
            + .25*einsum('ijab,ijab', t2, eris.oovv)
            + .5*einsum('ia,jb,ijab', t1, t1, eris.oovv))


def packed_correlation_energy(vector, space, eris):
    """Contract x2=2*t2(independent) without expanding the dense T2 tensor.

    Keep all four signed integral entries: raw integral derivatives need not
    obey the physical antisymmetries. Promote each gather before subtraction,
    matching the dense contraction when amplitudes and integrals differ in dtype.
    """
    no, nv = sum(space.nocc), sum(space.nvir)
    singles, doubles = space.indices
    ns = singles.shape[1]
    t1 = np.zeros((no, nv), vector.dtype).at[tuple(singles)].set(vector[:ns])
    i, j, a, b = doubles
    dtype = np.result_type(vector, eris.oovv)
    g = eris.oovv
    selected = (g[i, j, a, b].astype(dtype)-g[j, i, a, b].astype(dtype)
                -g[i, j, b, a].astype(dtype)+g[j, i, b, a].astype(dtype))
    return (einsum('ia,ia', eris.fock[:no, no:], t1)
            + .25*einsum('p,p', vector[ns:]/2, selected)
            + .5*einsum('ia,jb,ijab', t1, t1, eris.oovv))
