# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE.pyscf).
# Adapted from PySCF 2.9.0; see NOTICE.md for attribution and changes.
"""Spin-orbital CCSD intermediates, real in-core JAX contractions."""
from math import prod

import jax.numpy as np
from jax.numpy import einsum
from ._spin_virtual import _SpinVirtualERI

def make_tau(t2, t1a, t1b, fac=1, out=None):
    t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae',fov, t1)
    Fae += einsum('mf,amef->ae', t1, eris_vovv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fmi = ( foo + 0.5*einsum('me,ie->mi',fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fme = fov + einsum('nf,mnef->me', t1, eris.oovv)
    return Fme

def cc_Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris, *, tau=None):
    if isinstance(eris.vvvv, _SpinVirtualERI):
        raise NotImplementedError('A dense spin Wvvvv requires a dense bare virtual input')
    if tau is None:
        tau = make_tau(t2, t1, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau, 0.25*np.asarray(eris.oovv))
    return Wabef


def contract_Wvvvv(t1, t2, eris, x, *, tau=None):
    """Apply W to (..., v, v) without its large dressed-vvvv intermediate.

    The ovvv/oovv dressings contract with x first; their workspace scales as
    batch*o*v and batch*o**2. Bare vvvv is dense or a physical-block action. For
    occupied-dominated spaces, the original transient W can be smaller.
    Neither path assumes permutation symmetry of raw integrals or amplitudes.
    """
    nocc, nvir = t1.shape
    x = np.asarray(x)
    batch = prod(x.shape[:-2])
    compact = isinstance(eris.vvvv, _SpinVirtualERI)
    if not compact and 3*nvir**4 < batch*(2*nocc*nvir+nocc**2):
        return einsum('abef,...ef->...ab', cc_Wvvvv(t1, t2, eris, tau=tau), x)
    if tau is None:
        tau = make_tau(t2, t1, t1)
    result = (eris.vvvv.contract(x, dtype=eris.fock.dtype) if compact else
              einsum('abef,...ef->...ab', eris.vvvv, x))
    contracted = einsum('mbef,...ef->...mb', eris.ovvv, x)
    result -= einsum('ma,...mb->...ab', t1, contracted)
    result += einsum('mb,...ma->...ab', t1, contracted)
    contracted = einsum('mnef,...ef->...mn', eris.oovv, x)
    return result + 0.25*einsum('mnab,...mn->...ab', tau, contracted)

def cc_Wovvo(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej
