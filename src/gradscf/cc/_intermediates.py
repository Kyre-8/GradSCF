#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Modified for GradSCF: pure-JAX contractions, residual output, real MO blocks.
# Source: PySCF 2.9.0 (Apache-2.0); see NOTICE.md and LICENSE.pyscf.

from functools import partial
from math import prod
import jax.numpy as np

einsum = partial(np.einsum, precision="highest")


def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc, :nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki = 2 * einsum("kcld,ilcd->ki", eris_ovov, t2)
    Fki -= einsum("kdlc,ilcd->ki", eris_ovov, t2)
    Fki += 2 * einsum("kcld,ic,ld->ki", eris_ovov, t1, t1)
    Fki -= einsum("kdlc,ic,ld->ki", eris_ovov, t1, t1)
    Fki += foo
    return Fki


def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:, nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac = -2 * einsum("kcld,klad->ac", eris_ovov, t2)
    Fac += einsum("kdlc,klad->ac", eris_ovov, t2)
    Fac -= 2 * einsum("kcld,ka,ld->ac", eris_ovov, t1, t1)
    Fac += einsum("kdlc,ka,ld->ac", eris_ovov, t1, t1)
    Fac += fvv
    return Fac


def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc = 2 * einsum("kcld,ld->kc", eris_ovov, t1)
    Fkc -= einsum("kdlc,ld->kc", eris_ovov, t1)
    Fkc += fov
    return Fkc


def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    Lki = cc_Foo(t1, t2, eris) + einsum("kc,ic->ki", fov, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2 * einsum("lcki,lc->ki", eris_ovoo, t1)
    Lki -= einsum("kcli,lc->ki", eris_ovoo, t1)
    return Lki


def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    Lac = cc_Fvv(t1, t2, eris) - einsum("kc,ka->ac", fov, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2 * einsum("kdac,kd->ac", eris_ovvv, t1)
    Lac -= einsum("kcad,kd->ac", eris_ovvv, t1)
    return Lac


def cc_Woooo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij = einsum("lcki,jc->klij", eris_ovoo, t1)
    Wklij += einsum("kclj,ic->klij", eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wklij += einsum("kcld,ijcd->klij", eris_ovov, t2)
    Wklij += einsum("kcld,ic,jd->klij", eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def cc_Wvoov(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic = einsum("kcad,id->akic", eris_ovvv, t1)
    Wakic -= einsum("kcli,la->akic", eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
    eris_ovov = np.asarray(eris.ovov)
    Wakic -= 0.5 * einsum("ldkc,ilda->akic", eris_ovov, t2)
    Wakic -= 0.5 * einsum("lckd,ilad->akic", eris_ovov, t2)
    Wakic -= einsum("ldkc,id,la->akic", eris_ovov, t1, t1)
    Wakic += einsum("ldkc,ilad->akic", eris_ovov, t2)
    return Wakic


def cc_Wvovo(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci = einsum("kdac,id->akci", eris_ovvv, t1)
    Wakci -= einsum("lcki,la->akci", eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2, 0, 3, 1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5 * einsum("lckd,ilda->akci", eris_ovov, t2)
    Wakci -= einsum("lckd,id,la->akci", eris_ovov, t1, t1)
    return Wakci


def contract_Wvvvv(t1, eris, x):
    """Apply the restricted virtual intermediate with the smaller scratch.

    ``x`` is a dense ``(..., nvir, nvir)`` array, or an internal tuple (L, R)
    encoding ``x[i,j,c,d] = L[i,c] * R[j,d]`` with two independent matrices.
    Large virtual spaces contract each ovvv dressing with x into
    ``(..., nocc, nvir)`` before multiplying t1.
    If three virtual arrays cost less than that scratch, use a transient W
    only for this action. This static shape choice never caches a dressing;
    the bare chemists' vvvv input remains dense.
    """
    nocc, nvir = t1.shape
    factored = isinstance(x, tuple)
    if factored:
        left, right = map(np.asarray, x)
        batch = left.shape[0]*right.shape[0]
    else:
        x = np.asarray(x)
        batch = prod(x.shape[:-2])
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_vvvv = np.asarray(eris.vvvv)
    if 3*nvir**4 < batch*nocc*nvir:
        dressed = einsum("kcbd,ka->abcd", eris_ovvv, -t1)
        # D[b,a,d,c] is the other original dressing; no ERI symmetry is used.
        dressed += dressed.transpose(1, 0, 3, 2)
        dressed += eris_vvvv.transpose(0, 2, 1, 3)
        if factored:
            return einsum("abcd,ic,jd->ijab", dressed, left, right)
        return einsum("abcd,...cd->...ab", dressed, x)
    if factored:
        x = einsum("ic,jd->ijcd", left, right)
    result = einsum("acbd,...cd->...ab", eris_vvvv, x)
    contracted = einsum("kdac,...cd->...ka", eris_ovvv, x)
    result -= einsum("...ka,kb->...ab", contracted, t1)
    contracted = einsum("kcbd,...cd->...kb", eris_ovvv, x)
    return result - einsum("...kb,ka->...ab", contracted, t1)
