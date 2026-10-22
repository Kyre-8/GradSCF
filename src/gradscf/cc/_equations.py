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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#


# Modified for GradSCF: pure-JAX contractions, residual output, real MO blocks.
# Source: PySCF 2.9.0 (Apache-2.0); see NOTICE.md and LICENSE.pyscf.

from functools import partial
import jax.numpy as np
from . import _intermediates as imd

einsum = partial(np.einsum, precision="highest")


def residual_full(t1, t2, eris, *, cc2=False, doubles=True):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:]

    fov = fock[:nocc, nocc:].copy()
    foo = fock[:nocc, :nocc].copy()
    fvv = fock[nocc:, nocc:].copy()

    Foo = imd.cc_Foo(t1, t2, eris)
    Fvv = imd.cc_Fvv(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Foo = Foo.at[np.diag_indices(nocc)].add(-mo_e_o)
    Fvv = Fvv.at[np.diag_indices(nvir)].add(-mo_e_v)

    # T1 equation
    t1new = -2 * einsum("kc,ka,ic->ia", fov, t1, t1)
    t1new += einsum("ac,ic->ia", Fvv, t1)
    t1new += -einsum("ki,ka->ia", Foo, t1)
    t1new += 2 * einsum("kc,kica->ia", Fov, t2)
    t1new += -einsum("kc,ikca->ia", Fov, t2)
    t1new += einsum("kc,ic,ka->ia", Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2 * einsum("kcai,kc->ia", eris.ovvo, t1)
    t1new += -einsum("kiac,kc->ia", eris.oovv, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += 2 * einsum("kdac,ikcd->ia", eris_ovvv, t2)
    t1new += -einsum("kcad,ikcd->ia", eris_ovvv, t2)
    t1new += 2 * einsum("kdac,kd,ic->ia", eris_ovvv, t1, t1)
    t1new += -einsum("kcad,kd,ic->ia", eris_ovvv, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    t1new += -2 * einsum("lcki,klac->ia", eris_ovoo, t2)
    t1new += einsum("kcli,klac->ia", eris_ovoo, t2)
    t1new += -2 * einsum("lcki,lc,ka->ia", eris_ovoo, t1, t1)
    t1new += einsum("kcli,lc,ka->ia", eris_ovoo, t1, t1)

    if not doubles:
        eia = mo_e_o[:, None] - mo_e_v
        return t1new - eia * t1, np.zeros_like(t2)

    # T2 equation
    tmp2 = einsum("kibc,ka->abic", eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
    tmp = einsum("abic,jc->ijab", tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = einsum("kcai,jc->akij", eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = einsum("akij,kb->ijab", tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)
    if cc2:
        Woooo2 = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()
        Woooo2 += einsum("lcki,jc->klij", eris_ovoo, t1)
        Woooo2 += einsum("kclj,ic->klij", eris_ovoo, t1)
        Woooo2 += einsum("kcld,ic,jd->klij", eris.ovov, t1, t1)
        t2new += einsum("klij,ka,lb->ijab", Woooo2, t1, t1)
        t2new += imd.contract_Wvvvv(t1, eris, (t1, t1))
        Lvv2 = fvv - einsum("kc,ka->ac", fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = einsum("ac,ijcb->ijab", Lvv2, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)
        Loo2 = foo + einsum("kc,ic->ki", fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = einsum("ki,kjab->ijab", Loo2, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo = Loo.at[np.diag_indices(nocc)].add(-mo_e_o)
        Lvv = Lvv.at[np.diag_indices(nvir)].add(-mo_e_v)

        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        tau = t2 + einsum("ia,jb->ijab", t1, t1)
        t2new += einsum("klij,klab->ijab", Woooo, tau)
        t2new += imd.contract_Wvvvv(t1, eris, tau)
        tmp = einsum("ac,ijcb->ijab", Lvv, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)
        tmp = einsum("ki,kjab->ijab", Loo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)
        tmp = 2 * einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -= einsum("akci,kjcb->ijab", Wvovo, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)
        tmp = einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)
        tmp = einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    eia = mo_e_o[:, None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1new -= eia * t1
    t2new -= eijab * t2

    return t1new, t2new
