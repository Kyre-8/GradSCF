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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jun Yang
#

"""Real JAX CCSD density contractions adapted from PySCF 2.9.0.

See NOTICE.md for source attribution and local changes.
"""
import jax.numpy as jnp
einsum = jnp.einsum

def _gamma1( t1, t2, l1, l2):
    doo  =-einsum('ie,je->ij', l1, t1)
    doo -= einsum('imef,jmef->ij', l2, t2) * .5

    dvv  = einsum('ma,mb->ab', t1, l1)
    dvv += einsum('mnea,mneb->ab', t2, l2) * .5

    xt1  = einsum('mnef,inef->mi', l2, t2) * .5
    xt2  = einsum('mnfa,mnfe->ae', t2, l2) * .5
    xt2 += einsum('ma,me->ae', t1, l1)
    dvo  = einsum('imae,me->ai', t2, l1)
    dvo -= einsum('mi,ma->ai', xt1, t1)
    dvo -= einsum('ie,ae->ai', t1, xt2)
    dvo += t1.T

    dov = l1

    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
# When computing intermediates, the convention
# dm2[q,p,s,r] = <p^\dagger r^\dagger s q> is assumed in this function.
# It changes to dm2[p,q,r,s] = <p^\dagger r^\dagger s q> in _make_rdm2
def _gamma2(t1, t2, l1, l2, *, _einsum=None):
    einsum = jnp.einsum if _einsum is None else _einsum
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    miajb = einsum('ikac,kjcb->iajb', l2, t2)

    goovv = 0.25 * (l2.conj() + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
    tmp = einsum('ldjd->lj', miajb)
    goovv -= einsum('lj,liba->ijab', tmp, tau) * .25
    tmp = einsum('ldlb->db', miajb)
    goovv -= einsum('db,jida->ijab', tmp, tau) * .25
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, tau)
    goovv = goovv.conj()

    gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
    goooo = einsum('klab,ijab->klij', l2, tau) * 0.125

    gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
    gooov += einsum('iljk,la->jkia', goooo, t1)
    tmp = einsum('icjc->ij', miajb) * .25
    gooov -= einsum('ij,ka->jkia', tmp, t1)
    gooov += einsum('icja,kc->jkia', miajb, t1) * .5
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2, t1) * .25

    govvo  = einsum('ia,jb->ibaj', l1, t1)
    govvo += einsum('iajb->ibaj', miajb)
    govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)

    govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
    govvv += einsum('bcad,id->iabc', gvvvv, t1)
    tmp = einsum('kakb->ab', miajb) * .25
    govvv += einsum('ab,ic->iacb', tmp, t1)
    govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2, t1) * .25

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)


def _assemble_density_parts(one, two, dtype, *, zeros2=jnp.zeros):
    doo, dov, dvo, dvv = one
    no, nv = dov.shape
    o, v = slice(None, no), slice(no, None)
    dm1 = jnp.zeros((no+nv, no+nv), dtype)
    dm1 = dm1.at[o, o].set(doo+doo.T).at[o, v].set(dov+dvo.T)
    dm1 = dm1.at[v, o].set(dov.T+dvo).at[v, v].set(dvv+dvv.T) * .5
    dovov, dvvvv, doooo, _, dovvo, _, dovvv, dooov = two
    dm2 = zeros2((no+nv,)*4, dtype)
    dm2 = dm2.at[o, v, o, v].set(dovov)
    dm2 = dm2.at[v, o, v, o].set(dovov.transpose(1, 0, 3, 2))
    dm2 = dm2.at[o, o, v, v].set(-dovvo.transpose(0, 3, 2, 1))
    dm2 = dm2.at[v, v, o, o].set(-dovvo.transpose(2, 1, 0, 3))
    dm2 = dm2.at[o, v, v, o].set(dovvo)
    dm2 = dm2.at[v, o, o, v].set(dovvo.transpose(1, 0, 3, 2))
    dm2 = dm2.at[v, v, v, v].set(dvvvv).at[o, o, o, o].set(doooo)
    dm2 = dm2.at[o, v, v, v].set(dovvv)
    dm2 = dm2.at[v, v, o, v].set(dovvv.transpose(2, 3, 0, 1))
    dm2 = dm2.at[v, v, v, o].set(dovvv.transpose(3, 2, 1, 0))
    dm2 = dm2.at[v, o, v, v].set(dovvv.transpose(1, 0, 3, 2))
    dm2 = dm2.at[o, o, o, v].set(dooov)
    dm2 = dm2.at[o, v, o, o].set(dooov.transpose(2, 3, 0, 1))
    dm2 = dm2.at[o, o, v, o].set(dooov.transpose(1, 0, 3, 2))
    dm2 = dm2.at[v, o, o, o].set(dooov.transpose(3, 2, 1, 0))
    return dm1, dm2.transpose(1, 0, 3, 2)


def active_density_parts(t1, t2, l1, l2):
    """Original dense active density; retained as a small-system oracle."""
    return _assemble_density_parts(_gamma1(t1,t2,l1,l2),
        _gamma2(t1,t2,l1,l2),t1.dtype)


def indexed_density_parts(t1,t2,l1,l2):
    """The same density equations and O/V assignments with indexed outputs."""
    from ._density_index import Tensor,Buffer,einsum as indexed_einsum
    two = _gamma2(*map(Tensor,(t1,t2,l1,l2)),_einsum=indexed_einsum)
    return _assemble_density_parts(_gamma1(t1,t2,l1,l2),two,t1.dtype,zeros2=Buffer)


def reference_density_terms(p,dc,*,contract=jnp.einsum):
    """Original full-frame reference and core-active restoration terms."""
    return (contract('pq,rs->pqrs',p,dc)+contract('pq,rs->pqrs',dc,p)
            -contract('ps,rq->pqrs',p,dc)-contract('ps,rq->pqrs',dc,p)
            +contract('pq,rs->pqrs',p,p)-contract('ps,rq->pqrs',p,p))
