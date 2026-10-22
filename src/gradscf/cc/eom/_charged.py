# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.pyscf.
# Adapted from PySCF 2.9.0 pyscf/cc/eom_rccsd.py. See ../NOTICE.md.
"""Restricted IP/EA-CCSD actions; no partition approximation or solver code."""
from types import SimpleNamespace
from typing import NamedTuple
import jax.numpy as jnp
from . import _intermediates as imd


class _VirtualBlocks(NamedTuple):
    ovov: object
    ovvv: object
    vvvv: object

    def get_ovvv(self):
        return self.ovvv


def intermediates(t1, t2, ints, sector):
    common = dict(
        Loo=imd.Loo(t1, t2, ints),
        Lvv=imd.Lvv(t1, t2, ints),
        Fov=imd.cc_Fov(t1, t2, ints),
        Wovov=imd.Wovov(t1, t2, ints),
        Wovvo=imd.Wovvo(t1, t2, ints),
        Woovv=ints.ovov.transpose(0, 2, 1, 3),
        t2=t2,
    )
    if sector == "ip":
        common.update(
            Woooo=imd.Woooo(t1, t2, ints),
            Wooov=imd.Wooov(t1, t2, ints),
            Wovoo=imd.Wovoo(t1, t2, ints),
        )
    else:
        common.update(
            Wvovv=imd.Wvovv(t1, t2, ints),
            Wvvvo=imd.Wvvvo(t1, t2, ints, include_vvvv=False),
            virtual=(t1, t2, _VirtualBlocks(ints.ovov, ints.ovvv, ints.vvvv)),
        )
    return SimpleNamespace(**common)


def ip_action(r1, r2, w):
    einsum = jnp.einsum
    s1 = (
        -einsum("ki,k->i", w.Loo, r1)
        + 2 * einsum("ld,ild->i", w.Fov, r2)
        - einsum("kd,kid->i", w.Fov, r2)
    )
    s1 += -2 * einsum("klid,kld->i", w.Wooov, r2) + einsum("lkid,kld->i", w.Wooov, r2)
    s2 = -einsum("kbij,k->ijb", w.Wovoo, r1)
    s2 += (
        einsum("bd,ijd->ijb", w.Lvv, r2)
        - einsum("ki,kjb->ijb", w.Loo, r2)
        - einsum("lj,ilb->ijb", w.Loo, r2)
    )
    s2 += einsum("klij,klb->ijb", w.Woooo, r2) + 2 * einsum(
        "lbdj,ild->ijb", w.Wovvo, r2
    )
    s2 -= (
        einsum("kbdj,kid->ijb", w.Wovvo, r2)
        + einsum("lbjd,ild->ijb", w.Wovov, r2)
        + einsum("kbid,kjd->ijb", w.Wovov, r2)
    )
    tmp = 2 * einsum("lkdc,kld->c", w.Woovv, r2) - einsum("kldc,kld->c", w.Woovv, r2)
    s2 -= einsum("c,ijcb->ijb", tmp, w.t2)
    return s1, s2


def ea_action(r1, r2, w):
    einsum = jnp.einsum
    s1 = (
        einsum("ac,c->a", w.Lvv, r1)
        + 2 * einsum("ld,lad->a", w.Fov, r2)
        - einsum("ld,lda->a", w.Fov, r2)
    )
    s1 += einsum("alcd,lcd->a", 2 * w.Wvovv - w.Wvovv.transpose(0, 1, 3, 2), r2)
    s2 = einsum("abcj,c->jab", w.Wvvvo, r1)
    s2 += (
        einsum("ac,jcb->jab", w.Lvv, r2)
        + einsum("bd,jad->jab", w.Lvv, r2)
        - einsum("lj,lab->jab", w.Loo, r2)
    )
    s2 += einsum("lbdj,lad->jab", 2 * w.Wovvo - w.Wovov.transpose(0, 1, 3, 2), r2)
    s2 -= einsum("lajc,lcb->jab", w.Wovov, r2) + einsum("lbcj,lca->jab", w.Wovvo, r2)
    if hasattr(w, "virtual"):
        t1, t2, ints = w.virtual
        # Wvvvo's omitted W*t1 term shares this same virtual action. Keeping
        # t1 in the live RHS also retains its derivative when t1 is zero.
        s2 += imd.contract_Wvvvv(t1, t2, ints, r2+einsum("jd,c->jcd", t1, r1))
    else:
        # Explicit small-system namespace retained as the dense oracle.
        s2 += einsum("abcd,jcd->jab", w.Wvvvv, r2)
    tmp = einsum("klcd,lcd->k", 2 * w.Woovv - w.Woovv.transpose(0, 1, 3, 2), r2)
    s2 -= einsum("k,kjab->jab", tmp, w.t2)
    return s1, s2
