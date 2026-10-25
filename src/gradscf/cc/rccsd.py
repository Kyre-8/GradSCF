"""Restricted CC/QCI physical equations. Numerical forward/backward lives in solvers.

Full and CC2 residual contractions are adapted in _equations.py under Apache-2.0.
LCC models retain the constant and linear terms of the connected CC residual;
CCD restricts the cluster operator to doubles, without imposing a singles equation.
QCISD selects its own quadratic residual in _qci_equations.py and tau=t2 energy.
"""

import jax
import jax.numpy as jnp
from ._equations import residual_full
from ._qci_equations import residual_qcisd


def residual(t1, t2, ints, *, model="ccsd"):
    model = model.lower()
    if model == "qcisd":
        return residual_qcisd(t1, t2, ints)
    if model in {"ccd", "lccd"}:
        t1 = jnp.zeros_like(t1)
    if model == "ccs":
        t2 = jnp.zeros_like(t2)
    if model in {"lccd", "lccsd"}:
        zero = (jnp.zeros_like(t1), jnp.zeros_like(t2))
        r0, linear = jax.jvp(lambda a, b: residual_full(a, b, ints), zero, (t1, t2))
        r1, r2 = jax.tree.map(lambda a, b: a + b, r0, linear)
    else:
        r1, r2 = residual_full(t1, t2, ints, cc2=model == "cc2", doubles=model != "ccs")
    if model in {"ccd", "lccd"}:
        r1 = jnp.zeros_like(r1)
    if model == "ccs":
        r2 = jnp.zeros_like(r2)
    return r1, r2


def correlation_energy(t1, t2, ints, *, model="ccsd"):
    no = t1.shape[0]
    tau = t2 if model in {"lccd", "lccsd", "qcisd"} else t2 + jnp.einsum("ia,jb->ijab", t1, t1)
    return (
        2 * jnp.einsum("ia,ia->", ints.fock[:no, no:], t1)
        + 2 * jnp.einsum("ijab,iajb->", tau, ints.ovov)
        - jnp.einsum("ijab,ibja->", tau, ints.ovov)
    )
