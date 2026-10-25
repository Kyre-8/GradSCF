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

"""JAX charged-sector intermediates adapted from PySCF 2.9.0. See ../NOTICE.md."""


import jax.numpy as jnp
from .._intermediates import Loo, Lvv, cc_Fov, contract_Wvvvv as contract_virtual


def W1ovvo(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Wkaci = 2 * jnp.einsum("kcld,ilad->kaci", eris_ovov, t2)
    Wkaci += -jnp.einsum("kcld,liad->kaci", eris_ovov, t2)
    Wkaci += -jnp.einsum("kdlc,ilad->kaci", eris_ovov, t2)
    Wkaci += jnp.asarray(eris.ovvo).transpose(0, 2, 1, 3)
    return Wkaci


def W2ovvo(t1, t2, eris):
    Wkaci = jnp.einsum("la,lkic->kaci", -t1, Wooov(t1, t2, eris))
    eris_ovvv = jnp.asarray(eris.get_ovvv())
    Wkaci += jnp.einsum("kcad,id->kaci", eris_ovvv, t1)
    return Wkaci


def W1ovov(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Wkbid = -jnp.einsum("kcld,ilcb->kbid", eris_ovov, t2)
    Wkbid += jnp.asarray(eris.oovv).transpose(0, 2, 1, 3)
    return Wkbid


def W2ovov(t1, t2, eris):
    Wkbid = jnp.einsum("klid,lb->kbid", Wooov(t1, t2, eris), -t1)
    eris_ovvv = jnp.asarray(eris.get_ovvv())
    Wkbid += jnp.einsum("kcbd,ic->kbid", eris_ovvv, t1)
    return Wkbid


def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci


def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)


def Wooov(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Wklid = jnp.einsum("ic,kcld->klid", t1, eris_ovov)
    Wklid += jnp.asarray(eris.ovoo).transpose(2, 0, 3, 1)
    return Wklid


def Wovoo(t1, t2, eris):
    eris_ovoo = jnp.asarray(eris.ovoo)
    eris_ovvv = jnp.asarray(eris.get_ovvv())
    Wkbij = jnp.einsum("kbid,jd->kbij", W1ovov(t1, t2, eris), t1)
    Wkbij += -jnp.einsum("klij,lb->kbij", Woooo(t1, t2, eris), t1)
    Wkbij += jnp.einsum("kbcj,ic->kbij", W1ovvo(t1, t2, eris), t1)
    Wkbij += 2 * jnp.einsum("ldki,ljdb->kbij", eris_ovoo, t2)
    Wkbij += -jnp.einsum("ldki,jldb->kbij", eris_ovoo, t2)
    Wkbij += -jnp.einsum("kdli,ljdb->kbij", eris_ovoo, t2)
    Wkbij += jnp.einsum("kcbd,jidc->kbij", eris_ovvv, t2)
    Wkbij += jnp.einsum("kcbd,jd,ic->kbij", eris_ovvv, t1, t1)
    Wkbij += -jnp.einsum("kclj,libc->kbij", eris_ovoo, t2)
    Wkbij += jnp.einsum("kc,ijcb->kbij", cc_Fov(t1, t2, eris), t2)
    Wkbij += jnp.asarray(eris_ovoo).transpose(3, 1, 2, 0).conj()
    return Wkbij


def Woooo(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Wklij = jnp.einsum("kcld,ijcd->klij", eris_ovov, t2)
    Wklij += jnp.einsum("kcld,ic,jd->klij", eris_ovov, t1, t1)
    eris_ovoo = jnp.asarray(eris.ovoo)
    Wklij += jnp.einsum("ldki,jd->klij", eris_ovoo, t1)
    Wklij += jnp.einsum("kclj,ic->klij", eris_ovoo, t1)
    Wklij += jnp.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def Wvovv(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Walcd = jnp.einsum("ka,kcld->alcd", -t1, eris_ovov)
    Walcd += jnp.asarray(eris.get_ovvv()).transpose(2, 0, 3, 1)
    return Walcd


def Wvvvv(t1, t2, eris):
    eris_ovov = jnp.asarray(eris.ovov)
    Wabcd = jnp.einsum("kcld,klab->abcd", eris_ovov, t2)
    Wabcd += jnp.einsum("kcld,ka,lb->abcd", eris_ovov, t1, t1)
    Wabcd += jnp.asarray(eris.vvvv).transpose(0, 2, 1, 3)
    eris_ovvv = jnp.asarray(eris.get_ovvv())
    Wabcd -= jnp.einsum("ldac,lb->abcd", eris_ovvv, t1)
    Wabcd -= jnp.einsum("kcbd,ka->abcd", eris_ovvv, t1)
    return Wabcd


def contract_Wvvvv(t1, t2, eris, x):
    """Apply charged W without storing its dressed four-virtual tensor.

    The ground-state action supplies the same bare and ovvv terms. Only the
    charged t2/t1-t1 dressing remains, factored through two occupied indices.
    """
    result = contract_virtual(t1, eris, x)
    occupied = jnp.einsum("kcld,...cd->...kl", eris.ovov, x)
    result += jnp.einsum("...kl,klab->...ab", occupied, t2)
    return result+jnp.einsum("...kl,ka,lb->...ab", occupied, t1, t1)


def Wvvvo(t1, t2, eris, _Wvvvv=None, *, include_vvvv=True):
    nocc, nvir = t1.shape
    eris_ovvv = jnp.asarray(eris.get_ovvv())
    Wabcj = -jnp.einsum("alcj,lb->abcj", W1ovov(t1, t2, eris).transpose(1, 0, 3, 2), t1)
    Wabcj += -jnp.einsum("kbcj,ka->abcj", W1ovvo(t1, t2, eris), t1)
    Wabcj += 2 * jnp.einsum("ldac,ljdb->abcj", eris_ovvv, t2)
    Wabcj += -jnp.einsum("ldac,ljbd->abcj", eris_ovvv, t2)
    Wabcj += -jnp.einsum("lcad,ljdb->abcj", eris_ovvv, t2)
    Wabcj += -jnp.einsum("kcbd,jkda->abcj", eris_ovvv, t2)
    eris_ovoo = jnp.asarray(eris.ovoo)
    Wabcj += jnp.einsum("kclj,lkba->abcj", eris_ovoo, t2)
    Wabcj += jnp.einsum("kclj,lb,ka->abcj", eris_ovoo, t1, t1)
    Wabcj += -jnp.einsum("kc,kjab->abcj", cc_Fov(t1, t2, eris), t2)
    Wabcj += jnp.asarray(eris_ovvv).transpose(3, 1, 2, 0).conj()
    if include_vvvv:
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1, t2, eris)
        # Keep this term at t1=0: its parameter derivative need not vanish.
        Wabcj += jnp.einsum("abcd,jd->abcj", _Wvvvv, t1)
    return Wabcj
