"""Real molecular direct-RPA screening, with spectral primal/implicit response."""
from __future__ import annotations

from math import isqrt
import jax
import jax.numpy as jnp
from ..solvers.linear import factor_shifted, solve_shifted


def _rpa_matrices(mo_energy, b_ov, b_mn, *, eta=None):
    energy, ov, mn = map(jnp.asarray, (mo_energy, b_ov, b_mn))
    if energy.ndim != 1 or ov.ndim != 3 or mn.ndim != 3:
        raise ValueError('Expected energies (n,), OV factors (P,i,a) and MO factors (P,m,n)')
    if any(jnp.iscomplexobj(x) for x in (energy, ov, mn)):
        raise NotImplementedError('RPA resolvent requires real molecular inputs')
    nocc, nvirt = ov.shape[1:]
    if energy.size != nocc+nvirt or mn.shape[0] != ov.shape[0] or mn.shape[1] != mn.shape[2]:
        raise ValueError('Incompatible RPA energy/factor shapes')
    gaps = (energy[nocc:][None, :]-energy[:nocc, None]).reshape(-1)
    if eta is not None:
        gaps = gaps - 1j*jnp.asarray(eta)
    transition = ov.reshape(ov.shape[0], nocc*nvirt) * jnp.sqrt(gaps)[None, :]
    matrix = jnp.diag(gaps**2)+4*(transition.T @ transition)
    coupling = transition.T @ mn.reshape(mn.shape[0], mn.shape[1]**2)
    return matrix, coupling


def rpa_pole_expansion(mo_energy, b_ov, b_mn):
    """Reference pole representation; individual eigenbasis AD is not promised."""
    matrix, coupling = _rpa_matrices(mo_energy, b_ov, b_mn)
    values, vectors = jnp.linalg.eigh(matrix)
    nmo = b_mn.shape[1]
    return jnp.sqrt(values), (2*vectors.T @ coupling).reshape(values.size, nmo, nmo)


def screened_w_imag_poles(poles, couplings, frequencies):
    poles, couplings, frequencies = map(jnp.asarray, (poles, couplings, frequencies))
    weights = -1/(poles[None, :]**2+frequencies[:, None]**2)
    return jnp.einsum('wl,lmn->wmn', weights, couplings**2)


def screened_w_real_poles(poles, couplings, frequencies, *, eta=1e-3):
    poles, couplings, frequencies = map(jnp.asarray, (poles, couplings, frequencies))
    weights = -1/(poles[None, :]**2-(frequencies[:, None]+1j*eta)**2)
    return jnp.einsum('wl,lmn->wmn', weights, couplings**2)


def rpa_resolvent(mo_energy, b_ov, b_mn, *, eta=0.):
    """Prepare M and G once: Wc(z)[mn] = -4 G_mn.T (M-z^2 I)^-1 G_mn.

    The real-axis model preserves rho_response_real's asymmetric 2i*eta
    convention: d=gap-i*eta, z=omega+i*eta, M=diag(d**2)+4*T.T@T,
    T=B_ov*sqrt(d). Both models and the broadening are cached together.
    The returned MO-pair values are the diagonal of the pair-space quadratic
    form, not its full four-index matrix. Eigenvectors are cached exclusively
    for solving; the shared solver differentiates the matrix equation itself.
    """
    matrix, coupling = _rpa_matrices(mo_energy, b_ov, b_mn)
    model = dict(matrix=matrix, coupling=coupling, factorization=factor_shifted(matrix))
    if isinstance(eta, (int, float)) and eta == 0:
        retarded = dict(model)
    else:
        retarded_matrix, retarded_coupling = _rpa_matrices(mo_energy, b_ov, b_mn, eta=eta)
        retarded = dict(matrix=retarded_matrix, coupling=retarded_coupling,
                        factorization=factor_shifted(retarded_matrix))
    return dict(model, retarded=retarded, eta=jnp.asarray(eta))


def _screened_w_resolvent(model, shifts, pairs=None):
    coupling, state = model['coupling'], model['factorization']
    if pairs is not None:
        # One specified MO pair per frequency: no all-pairs response tensor.
        rhs = coupling[:, pairs].T
        def one(shift, b):
            return -4*jnp.sum(b*solve_shifted(state, b, shift))
        return jax.vmap(one)(shifts, rhs)
    def one(shift):
        return -4*jnp.sum(coupling*solve_shifted(state, coupling, shift), axis=0)
    flat = jax.vmap(one)(shifts)
    nmo = isqrt(coupling.shape[1])
    return flat.reshape((shifts.shape[0], nmo, nmo))


def screened_w_imag_resolvent(model, frequencies):
    return _screened_w_resolvent(model, -jnp.asarray(frequencies)**2)


def screened_w_real_resolvent(model, frequencies, *, pairs=None):
    return _screened_w_resolvent(model['retarded'],
        (jnp.asarray(frequencies)+1j*model['eta'])**2, pairs)
