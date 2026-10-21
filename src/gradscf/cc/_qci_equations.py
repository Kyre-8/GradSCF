# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Original author: Timothy Berkelbach <tim.berkelbach@gmail.com>
"""Real restricted QCISD residuals, adapted from PySCF 2.9.0 qcisd_slow.

Shared restricted intermediates are evaluated with zero singles where required
by QCI. Full Fock diagonals supply the physical residual directly; this module
contains no iteration, denominator division or differentiation rule.
See NOTICE.md for adaptation details and REFERENCES.md for method attribution.
"""
from functools import partial
import jax.numpy as jnp
from . import _intermediates as imd

einsum = partial(jnp.einsum, precision="highest")


def residual_qcisd(t1, t2, eris):
    no = t1.shape[0]
    zero = jnp.zeros_like(t1)
    foo = imd.cc_Foo(zero, t2, eris)
    fvv = imd.cc_Fvv(zero, t2, eris)
    fov = imd.cc_Fov(t1, t2, eris)
    ooov = eris.ovoo.transpose(2, 3, 0, 1)

    r1 = eris.fock[:no, no:]
    r1 = r1+einsum("ac,ic->ia", fvv, t1)-einsum("ki,ka->ia", foo, t1)
    r1 += 2*einsum("kc,kica->ia", fov, t2)-einsum("kc,ikca->ia", fov, t2)
    r1 += 2*einsum("kcai,kc->ia", eris.ovvo, t1)-einsum("kiac,kc->ia", eris.oovv, t1)
    r1 += 2*einsum("kdac,ikcd->ia", eris.ovvv, t2)-einsum("kcad,ikcd->ia", eris.ovvv, t2)
    r1 += -2*einsum("kilc,klac->ia", ooov, t2)+einsum("likc,klac->ia", ooov, t2)

    r2 = eris.ovov.transpose(0, 2, 1, 3)
    loo = imd.Loo(zero, t2, eris)
    lvv = imd.Lvv(zero, t2, eris)
    woooo = imd.cc_Woooo(zero, t2, eris)
    wvoov = imd.cc_Wvoov(zero, t2, eris)
    wvovo = imd.cc_Wvovo(zero, t2, eris)
    r2 = r2+einsum("klij,klab->ijab", woooo, t2)+imd.contract_Wvvvv(zero, eris, t2)
    tmp = einsum("ac,ijcb->ijab", lvv, t2)-einsum("ki,kjab->ijab", loo, t2)
    tmp += 2*einsum("akic,kjcb->ijab", wvoov, t2)-einsum("akci,kjcb->ijab", wvovo, t2)
    tmp -= einsum("akic,kjbc->ijab", wvoov, t2)+einsum("bkci,kjac->ijab", wvovo, t2)
    tmp += einsum("abic,jc->ijab", eris.ovvv.transpose(1, 3, 0, 2), t1)
    tmp -= einsum("akij,kb->ijab", ooov.transpose(3, 1, 2, 0), t1)
    return r1, r2+tmp+tmp.transpose(1, 0, 3, 2)
