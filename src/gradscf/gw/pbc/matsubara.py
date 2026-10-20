"""Full complex finite-k GW on finite-temperature imaginary axes.

All orbital matrices live in the fixed orthonormal source-MO frame. Density
is per spin; the closed-shell factor two and the Brillouin-zone 1/Nk occur
once in the bubble. Both signs of every Matsubara frequency are retained.
In particular, neither Pi nor W is assumed even, real or Hermitian at an
individual nonzero imaginary frequency.

Static-reference subtraction supplies exact thermal G and Pi. A signed
transition RPA spectral representation supplies W and G0 W analytically,
including the degenerate thermal zero mode. First odd bosonic moments and
the noncommuting self-energy endpoint moment are subtracted explicitly.
This is an eager forward implementation; no periodic scGW AD is claimed.
"""
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ...scf._pytree import pytree_dataclass
from ..matsubara import (MatsubaraGrid as _MolecularGrid,
                        matsubara_grid as _matsubara_grid, _reference_green)


@pytree_dataclass(static_fields=('beta', 'nw'))
@dataclass(frozen=True)
class MatsubaraGrid(_MolecularGrid):
    """PBC-only quadrature weights; molecular grids retain their original API."""
    tau_weights: jax.Array


def matsubara_grid(*, nw, beta, time_oversampling=2, time_quadrature='uniform'):
    """Keep the frequency window fixed while refining time integration.

    The default Nt=4*nw resolves quadratic products of the retained G/W
    modes before projection into the original frequency window. Reference
    functions have infinite bandwidth, so grid convergence remains required.
    time_oversampling=1 preserves the original Nt=2*nw/Nyquist-half baseline.
    On a padded grid both stored bosonic endpoints are interior modes and
    each has its full Matsubara-sum weight.
    Optional 'gauss' uses positive Gauss-Legendre weights on (0,beta).
    Symmetric nodes preserve G(beta-tau) by reversal. This quadrature does
    not change the inverse Matsubara sum or the retained frequency window;
    its finite Fourier and endpoint accuracy must still be checked.
    """
    if (isinstance(time_oversampling, bool) or not isinstance(time_oversampling, Integral)
            or time_oversampling < 1):
        raise ValueError('time_oversampling must be an integer >= 1')
    if time_quadrature not in ('uniform', 'gauss'):
        raise ValueError("time_quadrature must be 'uniform' or 'gauss'")
    grid = _matsubara_grid(nw=nw, beta=beta)
    ntau = len(grid.tau)*int(time_oversampling)
    if time_quadrature == 'gauss':
        node, weight = np.polynomial.legendre.leggauss(ntau)
        tau = jnp.asarray(.5*grid.beta*(node+1))
        tau_weights = jnp.asarray(.5*grid.beta*weight)
    else:
        tau = (jnp.arange(ntau, dtype=jnp.float64)+.5)*grid.beta/ntau
        tau_weights = jnp.full(ntau, grid.beta/ntau, dtype=jnp.float64)
    weights = (grid.boson_weights if time_quadrature == 'uniform' and time_oversampling == 1
               else jnp.ones_like(grid.boson_weights))
    return MatsubaraGrid(grid.beta, grid.nw, tau, grid.fermion, grid.boson, weights, tau_weights)


def _dagger(matrix):
    return matrix.conj().swapaxes(-1, -2)


def _reference(fock, mu, grid):
    density, giw, gtau = jax.vmap(lambda f: _reference_green(f, mu, grid))(fock)
    energy, coeff = jnp.linalg.eigh(fock)
    occupation = jax.nn.sigmoid(-grid.beta*(energy-mu))
    return energy, coeff, occupation, density, giw, gtau


def _moment_tolerance(eigenvalue):
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalue), axis=-1), jnp.finfo(eigenvalue.dtype).tiny)
    return 64*eigenvalue.shape[-1]*jnp.finfo(eigenvalue.dtype).eps*scale


def _moment_is_psd(moment):
    eigenvalue = jnp.linalg.eigvalsh(.5*(moment+_dagger(moment)))
    return jnp.all(eigenvalue >= -_moment_tolerance(eigenvalue)[..., None])


def _green_tail_reference(fock, sigma_last, mu, grid, sigma_moment):
    """Causal single-matrix-pole reference used ONLY for tail subtraction.

    M1=L L^dagger and S=L^+ Sigma(zmax) L^{+dagger}. The Hermitian
    A=Herm[zmax I-S^-1] defines a numerical reference L(z-A)^-1 L^dagger.
    Its augmented Hamiltonian gives exact reference density/G(tau). This
    leaves the supplied Dyson self-energy unchanged. It is exact for a
    single matrix pole, while general spectra still need grid convergence.
    Significant negative M1 returns nonfinite reference data, never a
    projected physical moment. Only eigenvalues within roundoff of zero
    are treated as numerical nullspace. The bath's mu*I term remains in
    the chemical-potential response; fitted A has no explicit mu dependence.
    Returns density, G(iw), G(tau), G'(0+), and G'(beta-).
    """
    def static(f):
        density, giw, gtau = _reference_green(f, mu, grid)
        shift = f-mu*jnp.eye(f.shape[-1], dtype=f.dtype)
        derivative_beta = shift@density
        return density, giw, gtau, shift-derivative_beta, derivative_beta

    if sigma_moment is None:
        return jax.vmap(static)(fock)
    # Match both lax.cond branches even for a real source Fock matrix.
    fock = jnp.asarray(fock, dtype=jnp.complex128)
    zmax = 1j*grid.fermion[-1]

    def one(f, sigma, moment):
        def dynamic(_):
            value, vector = jnp.linalg.eigh(.5*(moment+_dagger(moment)))
            tolerance = _moment_tolerance(value)
            valid = jnp.all(value >= -tolerance)
            supported = value > tolerance
            safe = jnp.where(supported, value, 1.)
            root = (vector*jnp.where(supported, jnp.sqrt(safe), 0.)[None])@_dagger(vector)
            inverse = (vector*jnp.where(supported, 1/jnp.sqrt(safe), 0.)[None])@_dagger(vector)
            projector = (vector*supported[None])@_dagger(vector)
            eye = jnp.eye(f.shape[-1], dtype=jnp.complex128)
            normalized = inverse@sigma@inverse+(eye-projector)/zmax
            bath = zmax*eye-jnp.linalg.inv(normalized)
            bath = .5*(bath+_dagger(bath))
            augmented = jnp.concatenate((jnp.concatenate((f, root), axis=1),
                jnp.concatenate((_dagger(root), mu*eye+bath), axis=1)), axis=0)
            density, giw, gtau = _reference_green(augmented, mu, grid)
            n = f.shape[-1]
            energy_density = (augmented-mu*jnp.eye(2*n, dtype=augmented.dtype))@density
            derivative_beta = energy_density[:n, :n]
            output = (density[:n, :n], giw[..., :n, :n], gtau[..., :n, :n],
                      f-mu*eye-derivative_beta, derivative_beta)
            return tuple(jnp.where(valid, item, jnp.nan) for item in output)
        return jax.lax.cond(jnp.any(moment != 0), dynamic,
                           lambda _: static(f), operand=None)

    return jax.vmap(one)(fock, sigma_last, sigma_moment)


@jax.jit
def dyson_green_and_density(fock, sigma_iw, mu, grid, sigma_moment=None):
    """Return G[k,iw,m,n] and the Hermitian ONE-SPIN density.

    The full complex Fourier correction is retained before Hermitian
    projection. Taking its elementwise real part would discard physical
    imaginary off-diagonal density in a Bloch orbital frame.
    Supplying the physical 1/z sigma_moment enables a causal numerical
    matrix-pole reference for both the even density tail and G(tau) tail.
    """
    eye = jnp.eye(fock.shape[-1], dtype=jnp.complex128)
    inverse = (1j*grid.fermion[None, :, None, None]+mu)*eye-fock[:, None]-sigma_iw
    green = jnp.linalg.solve(inverse, jnp.broadcast_to(eye, inverse.shape))
    density, reference, _, _, _ = _green_tail_reference(fock, sigma_iw[:, -1], mu, grid, sigma_moment)
    density = density+jnp.sum(green-reference, axis=1)/grid.beta
    return green, .5*(density+_dagger(density))


def _time_chunks(contract, operands, batch_size):
    """Evaluate one contraction with a leading independent time axis."""
    if batch_size is not None and (isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral) or batch_size < 1):
        raise ValueError('time_batch_size must be None or an integer >= 1')
    ntau = operands[0].shape[0]
    if batch_size is None or batch_size >= ntau:
        return contract(*operands)
    chunks = []
    for start in range(0, ntau, int(batch_size)):
        current = tuple(array[start:start+batch_size] for array in operands)
        chunks.append(jax.block_until_ready(contract(*current)))
    return jnp.concatenate(chunks, axis=0)


def _bubble(g_tau, factors, table, *, time_batch_size=None):
    nk = g_tau.shape[0]
    blocks = []
    for q, b in enumerate(factors):
        def contract(partner, reverse):
            return -2/nk*jnp.einsum('kPab,tkbc,kQdc,tkda->tPQ',
                                    b, partner, b.conj(), reverse)
        # Reversal is global before slicing; reversing each chunk would
        # incorrectly replace beta-tau with a local interval reflection.
        operands = (g_tau[np.asarray(table)[:, q]].swapaxes(0, 1),
                    g_tau[:, ::-1].swapaxes(0, 1))
        blocks.append(_time_chunks(contract, operands, time_batch_size))
    return tuple(blocks)


def _bubble_moment(density, factors, table):
    """Exact coefficient of 1/z, including noncommuting complex vertices."""
    nk = density.shape[0]
    return tuple(2/nk*(
        jnp.einsum('kab,kPbc,kQac->PQ', density, b, b.conj())
        -jnp.einsum('kPab,kbc,kQac->PQ', b, density[np.asarray(table)[:, q]], b.conj()))
        for q, b in enumerate(factors))


def _bubble_second_moment(density, derivative_zero, derivative_beta, factors, table):
    """Coefficient Pi2=Pi'(0+)-Pi'(beta-) in Pi(z)=Pi1/z+Pi2/z^2.

    G endpoints are rho-I and -rho. Matrix order is essential for complex
    projected density vertices; no commutation with rho or G' is assumed.
    """
    nk, nmo = density.shape[:2]
    eye = jnp.eye(nmo, dtype=density.dtype)
    blocks = []
    for q, b in enumerate(factors):
        partner = np.asarray(table)[:, q]
        trace = lambda x, y: jnp.einsum('kPab,kbc,kQdc,kda->PQ', b, x, b.conj(), y)
        block = 2/nk*(trace(derivative_zero[partner], density)
            +trace(density[partner]-eye, derivative_beta)
            +trace(derivative_beta[partner], density-eye)
            +trace(density[partner], derivative_zero))
        blocks.append(.5*(block+_dagger(block)))
    return tuple(blocks)


def _odd_reference(grid, scale):
    """r(z)=z/(z^2-a^2), with r(0+)=-1/2 and r(beta-)=+1/2."""
    riv = 1j*grid.boson/((1j*grid.boson)**2-scale**2)
    denominator = -jnp.expm1(-grid.beta*scale)
    rtau = (jnp.exp(-(grid.beta-grid.tau)*scale)-jnp.exp(-grid.tau*scale))/(2*denominator)
    return riv, rtau


def _even_reference(grid, scale):
    """s(z)=1/(z^2-a^2), inverse time value has equal negative endpoints."""
    siv = -1/(grid.boson**2+scale**2)
    denominator = 2*scale*(-jnp.expm1(-grid.beta*scale))
    stau = -(jnp.exp(-grid.tau*scale)+jnp.exp(-(grid.beta-grid.tau)*scale))/denominator
    return siv, stau


def _reference_pi(factors, table, energy, coeff, occupation, grid):
    blocks, rotated = [], []
    for q, b in enumerate(factors):
        partner = np.asarray(table)[:, q]
        vertex = jnp.einsum('kam,kPab,kbn->kPmn', coeff.conj(), b, coeff[partner])
        rotated.append(vertex)
        delta = energy[:, :, None]-energy[partner, None, :]
        numerator = occupation[:, :, None]-occupation[partner, None, :]
        denominator = 1j*grid.boson[:, None, None, None]+delta[None]
        degenerate = (jnp.abs(delta)[None] < 1e-12)&(grid.boson[:, None, None, None] == 0.)
        quotient = numerator[None]/jnp.where(degenerate, 1., denominator)
        mean_occupation = .5*(occupation[:, :, None]+occupation[partner, None, :])
        quotient = jnp.where(degenerate,
            -grid.beta*mean_occupation[None]*(1-mean_occupation[None]), quotient)
        blocks.append(2/energy.shape[0]*jnp.einsum('kPab,wkab,kQab->wPQ',
                                                vertex, quotient, vertex.conj()))
    return tuple(blocks), tuple(rotated)


def _screened(pi):
    eye = jnp.eye(pi.shape[-1], dtype=pi.dtype)
    return jnp.linalg.solve(eye-pi, pi)


def _boson_kernel(poles, time, beta):
    """Inverse transform of 1/(z-pole), stable for either pole sign."""
    absolute = jnp.abs(poles)
    denominator = -jnp.expm1(-beta*absolute)
    return jnp.where(poles[None] > 0,
        -jnp.exp(-time[:, None]*absolute)/denominator,
        jnp.exp(-(beta-time[:, None])*absolute)/denominator)


def _bose(poles, beta):
    small = jnp.exp(-beta*jnp.abs(poles))
    denominator = -jnp.expm1(-beta*jnp.abs(poles))
    return jnp.where(poles > 0, small/denominator, -1/denominator)


def _reference_screening(rotated, table, energy, occupation, pi_ref, grid):
    """Signed-transition RPA poles, with no real/even-W approximation.

    Pi=V(z-S D)^-1 S V^dagger. With A=D+V^dagger V>0,
    H=A^1/2 S A^1/2 is Hermitian and
    Wc=sum_s lambda_s a_s a_s^dagger/(z-lambda_s),
    a=V A^-1/2 U. Transitions with occupation weight below 1e-14
    contribute to the exact Pi array but are omitted from this tail-only
    reference; the finite-grid difference restores their resolved response.
    """
    references = []
    nk = energy.shape[0]
    for q, b in enumerate(rotated):
        partner = np.asarray(table)[:, q]
        difference = energy[partner, None, :]-energy[:, :, None]
        weight = occupation[:, :, None]-occupation[partner, None, :]
        columns = b.transpose(1, 0, 2, 3).reshape(b.shape[1], -1)
        selected = np.asarray((jnp.abs(difference).reshape(-1) > 1e-12)
            &(jnp.abs(weight).reshape(-1) > 1e-14)&jnp.any(columns != 0, axis=0))
        gaps = jnp.abs(difference.reshape(-1)[selected])
        signs = jnp.sign(weight.reshape(-1)[selected])
        vertex = columns[:, selected]
        vertex = vertex*jnp.sqrt(2/nk*jnp.abs(weight.reshape(-1)[selected]))[None]
        if gaps.size:
            aval, avec = jnp.linalg.eigh(jnp.diag(gaps)+_dagger(vertex)@vertex)
            if not bool(jnp.all(aval > 0)):
                raise ArithmeticError('Periodic reference RPA metric is not positive definite.')
            root = (avec*jnp.sqrt(aval)[None])@_dagger(avec)
            inverse_root = (avec/jnp.sqrt(aval)[None])@_dagger(avec)
            poles, modes = jnp.linalg.eigh((root*signs[None])@root)
            if not bool(jnp.all(jnp.abs(poles) > 1e-14)):
                raise ArithmeticError('Unresolved zero-energy periodic RPA pole.')
            amplitude = vertex@inverse_root@modes
            dynamic_iw = jnp.einsum('Ps,ws,Qs->wPQ', amplitude,
                poles[None]/(1j*grid.boson[:, None]-poles[None]), amplitude.conj())
            dynamic_zero = -amplitude@_dagger(amplitude)
            dynamic_moment = (amplitude*poles[None])@_dagger(amplitude)
            dynamic_second_moment = (amplitude*(poles**2)[None])@_dagger(amplitude)
            kernel = _boson_kernel(poles, grid.tau, grid.beta)*poles[None]
            dynamic_tau = jnp.einsum('Ps,ts,Qs->tPQ', amplitude, kernel, amplitude.conj())
            endpoints = _boson_kernel(poles, jnp.array([0., grid.beta]), grid.beta)*poles[None]
            endpoint = jnp.einsum('Ps,ts,Qs->tPQ', amplitude, endpoints, amplitude.conj())
        else:
            poles = jnp.zeros((0,), dtype=energy.dtype)
            amplitude = jnp.zeros((b.shape[1], 0), dtype=b.dtype)
            dynamic_iw = jnp.zeros_like(pi_ref[q])
            dynamic_zero = jnp.zeros_like(pi_ref[q][0])
            dynamic_moment = jnp.zeros_like(dynamic_zero)
            dynamic_second_moment = jnp.zeros_like(dynamic_zero)
            dynamic_tau = jnp.zeros((len(grid.tau), b.shape[1], b.shape[1]), dtype=b.dtype)
            endpoint = jnp.zeros((2, b.shape[1], b.shape[1]), dtype=b.dtype)
        exact_zero = _screened(pi_ref[q][grid.nw])
        # Degenerate/intraband thermal response is a frequency-zero term.
        static = (exact_zero-dynamic_zero)/grid.beta
        reference_iw = dynamic_iw.at[grid.nw].add(grid.beta*static)
        references.append(dict(iw=reference_iw, tau=dynamic_tau+static,
            zero=endpoint[0]+static, beta=endpoint[1]+static,
            poles=poles, amplitude=amplitude, static=static, moment=dynamic_moment,
            second_moment=dynamic_second_moment))
    return tuple(references)


def _sigma_tau(g_tau, factors, inverse, wc_tau, *, time_batch_size=None):
    nk = g_tau.shape[0]
    sigma = jnp.zeros_like(g_tau)
    for q, b in enumerate(factors):
        internal = np.asarray(inverse)[:, q]
        vertex = b[internal]
        def contract(green, screened):
            return -jnp.einsum('kPma,tkmn,kQnb,tPQ->tkab',
                               vertex.conj(), green, vertex, screened)/nk
        contribution = _time_chunks(contract,
            (g_tau[internal].swapaxes(0, 1), wc_tau[q]), time_batch_size)
        sigma = sigma+contribution.swapaxes(0, 1)
    return sigma


def _sigma_moment(density, factors, inverse, wc_zero, wc_beta):
    nk, nmo = density.shape[:2]
    moment = jnp.zeros_like(density, dtype=jnp.complex128)
    for q, b in enumerate(factors):
        internal = np.asarray(inverse)[:, q]
        vertex = b[internal]
        moment = moment-jnp.einsum('kPma,kmn,kQnb,PQ->kab',
            vertex.conj(), jnp.eye(nmo)-density[internal], vertex, wc_zero[q])/nk
        moment = moment-jnp.einsum('kPma,kmn,kQnb,PQ->kab',
            vertex.conj(), density[internal], vertex, wc_beta[q])/nk
    return .5*(moment+_dagger(moment))


def _reference_sigma(factors, inverse, energy, coeff, occupation, mu, giw, reference, grid):
    nk, nmo = energy.shape
    sigma = jnp.zeros((nk, len(grid.fermion), nmo, nmo), dtype=jnp.complex128)
    for q, b in enumerate(factors):
        internal = np.asarray(inverse)[:, q]
        vertex = jnp.einsum('kmi,kPma->kPia', coeff[internal].conj(), b[internal])
        poles, amplitude = reference[q]['poles'], reference[q]['amplitude']
        coupling = jnp.einsum('Ps,kPia->ksia', amplitude.conj(), vertex)
        weight = poles[None, None, None, :]*(
            1-occupation[internal, None, :, None]+_bose(poles, grid.beta)[None, None, None])/(
            1j*grid.fermion[None, :, None, None]-(energy[internal]-mu)[:, None, :, None]
            -poles[None, None, None, :])
        sigma = sigma+jnp.einsum('ksia,kwis,ksib->kwab', coupling.conj(), weight, coupling)/nk
        sigma = sigma-jnp.einsum('kPma,kwmn,kQnb,PQ->kwab', b[internal].conj(),
            giw[internal], b[internal], reference[q]['static'])/nk
    return sigma


def gw_matsubara_step(green_iw, fock, mu, model, grid, sigma_moment=None, *,
                       density=None, time_diagnostics=True, time_batch_size=None):
    """Build the full finite-k Pi, Wc and correlation Sigma from one G.

    ``density`` and ``sigma_moment`` describe this same input G. Omitting
    density uses the numerical-reference-subtracted fermionic sum. The
    bosonic endpoint values are returned separately because finite complex
    projected density vertices can have a nonzero odd high-frequency tail.
    Energies require a separate converged functional evaluation and are not
    inferred from static Fock eigenvalues here.

    Every q is evaluated by the same helper and consumed before the next q.
    ``time_diagnostics=False`` omits green_tau, screened_tau, screened_zero,
    screened_beta and sigma_tau from the returned dictionary. Frequency Pi/W
    and their moments are retained. Omitted diagnostics are absent, rather
    than represented by None. SC iterations use this bounded retention path.
    ``time_batch_size`` optionally bounds independent-time contraction
    workspaces; chunks are concatenated before the unchanged Fourier sum.
    """
    factors = model.correlation_factors
    physical = (green_iw, fock, mu, factors, sigma_moment, density)
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(physical)):
        raise NotImplementedError('Periodic matrix GW is eager; periodic scGW differentiation/AD is not implemented.')
    energy, coeff, occupation, rho_ref, giw_ref, gtau_ref = _reference(fock, mu, grid)
    eye = jnp.eye(fock.shape[-1], dtype=jnp.complex128)
    shift = fock-mu*eye
    derivative_beta_ref = shift@rho_ref
    derivative_zero_ref = shift-derivative_beta_ref
    if sigma_moment is None:
        density_tail, giw_tail, gtau_tail = rho_ref, giw_ref, gtau_ref
        derivative_beta_tail = derivative_beta_ref
    else:
        sigma_last = (1j*grid.fermion[-1]+mu)*eye-fock-jnp.linalg.inv(green_iw[:, -1])
        density_tail, giw_tail, gtau_tail, _, derivative_beta_tail = _green_tail_reference(
            fock, sigma_last, mu, grid, sigma_moment)
    residual_g = green_iw-giw_tail
    if density is None:
        density = density_tail+jnp.sum(residual_g, axis=1)/grid.beta
        density = .5*(density+_dagger(density))
    derivative_beta = derivative_beta_tail+jnp.einsum('w,kwab->kab',
        1j*grid.fermion, residual_g)/grid.beta
    derivative_beta = .5*(derivative_beta+_dagger(derivative_beta))
    derivative_zero = shift-derivative_beta
    phase_f = jnp.exp(-1j*grid.tau[:, None]*grid.fermion[None])
    gtau = gtau_tail+jnp.einsum('tw,kwab->ktab', phase_f, residual_g)/grid.beta
    scale = jnp.maximum(1., jnp.max(jnp.abs(energy-mu)))
    moments = _bubble_moment(density, factors, model.table)
    moments_ref = _bubble_moment(rho_ref, factors, model.table)
    second = _bubble_second_moment(density, derivative_zero, derivative_beta, factors, model.table)
    second_ref = _bubble_second_moment(rho_ref, derivative_zero_ref, derivative_beta_ref, factors, model.table)
    riv, rtau = _odd_reference(grid, scale)
    siv, stau = _even_reference(grid, scale)
    even_endpoint = -(1+jnp.exp(-grid.beta*scale))/(2*scale*(-jnp.expm1(-grid.beta*scale)))
    forward_b = jnp.exp(1j*grid.boson[:, None]*grid.tau[None])
    backward_b = forward_b.conj()
    tau_weights = getattr(grid, 'tau_weights',
                          jnp.full(len(grid.tau), grid.beta/len(grid.tau)))

    def one_q(q, factor):
        qfactors = (factor,)
        table = np.asarray(model.table)[:, q:q+1]
        inverse = np.asarray(model.inverse)[:, q:q+1]
        pi_ref, rotated = _reference_pi(qfactors, table, energy, coeff, occupation, grid)
        reference = _reference_screening(rotated, table, energy, occupation, pi_ref, grid)[0]
        # Synchronization bounds transient contraction buffers as well as
        # Python references; no large time array from another q remains live.
        bubble = jax.block_until_ready(_bubble(gtau, qfactors, table,
                                               time_batch_size=time_batch_size)[0])
        bubble_ref = _bubble(gtau_ref, qfactors, table,
                              time_batch_size=time_batch_size)[0]
        delta_moment = moments[q]-moments_ref[q]
        delta_second = second[q]-second_ref[q]
        difference = (bubble-bubble_ref-rtau[:, None, None]*delta_moment
                      -stau[:, None, None]*delta_second)
        block = (pi_ref[0]+jnp.einsum('wt,t,tPQ->wPQ', forward_b, tau_weights, difference)
                 +riv[:, None, None]*delta_moment+siv[:, None, None]*delta_second)
        block = jax.block_until_ready(block)
        del bubble, bubble_ref, difference
        screened = _screened(block)
        delta_wc_moment = moments[q]-reference['moment']
        wc_second = second[q]+moments[q]@moments[q]
        delta_wc_second = wc_second-reference['second_moment']
        difference = (screened-reference['iw']-riv[:, None, None]*delta_wc_moment
                      -siv[:, None, None]*delta_wc_second)
        wc_tau = (reference['tau']+jnp.einsum('wt,w,wPQ->tPQ',
            backward_b, grid.boson_weights, difference)/grid.beta
            +rtau[:, None, None]*delta_wc_moment+stau[:, None, None]*delta_wc_second)
        wc_tau = jax.block_until_ready(wc_tau)
        endpoint = jnp.einsum('w,wPQ->PQ', grid.boson_weights, difference)/grid.beta
        wc_zero = reference['zero']+endpoint-.5*delta_wc_moment+even_endpoint*delta_wc_second
        wc_beta = reference['beta']+endpoint+.5*delta_wc_moment+even_endpoint*delta_wc_second
        result = dict(pi=block, wc=screened, wc_second=wc_second,
            sigma_tau=_sigma_tau(gtau, qfactors, inverse, (wc_tau,),
                                  time_batch_size=time_batch_size),
            sigma_tau_ref=_sigma_tau(gtau_ref, qfactors, inverse, (reference['tau'],),
                                      time_batch_size=time_batch_size),
            moment=_sigma_moment(density, qfactors, inverse, (wc_zero,), (wc_beta,)),
            moment_ref=_sigma_moment(rho_ref, qfactors, inverse,
                                    (reference['zero'],), (reference['beta'],)),
            sigma_ref=_reference_sigma(qfactors, inverse, energy, coeff, occupation,
                                       mu, giw_ref, (reference,), grid))
        if time_diagnostics:
            result.update(wc_tau=wc_tau, wc_zero=wc_zero, wc_beta=wc_beta)
        return jax.block_until_ready(result)

    pi, wc, wc_second, wc_tau, wc_zero, wc_beta = [], [], [], [], [], []
    sigma_tau = jnp.zeros_like(gtau)
    sigma_tau_ref = jnp.zeros_like(gtau_ref)
    moment = jnp.zeros_like(density, dtype=jnp.complex128)
    moment_ref = jnp.zeros_like(moment)
    sigma_ref = jnp.zeros_like(green_iw)
    for q, factor in enumerate(factors):
        contribution = one_q(q, factor)
        pi.append(contribution['pi'])
        wc.append(contribution['wc'])
        wc_second.append(contribution['wc_second'])
        sigma_tau = sigma_tau+contribution['sigma_tau']
        sigma_tau_ref = sigma_tau_ref+contribution['sigma_tau_ref']
        moment = moment+contribution['moment']
        moment_ref = moment_ref+contribution['moment_ref']
        sigma_ref = sigma_ref+contribution['sigma_ref']
        if time_diagnostics:
            wc_tau.append(contribution['wc_tau'])
            wc_zero.append(contribution['wc_zero'])
            wc_beta.append(contribution['wc_beta'])
        del contribution
    delta_moment = moment-moment_ref
    sigma = sigma_ref+jnp.einsum('tw,t,ktab->kwab', phase_f.conj(), tau_weights,
        sigma_tau-sigma_tau_ref+.5*delta_moment[:, None])+delta_moment[:, None]/(1j*grid.fermion[None, :, None, None])
    result = dict(polarizability_inu=tuple(pi), screened_inu=tuple(wc),
        polarizability_second_moment=second, screened_second_moment=tuple(wc_second),
        sigma_iw=sigma, sigma_moment=moment, correlation_energy=None,
        time_diagnostics_retained=bool(time_diagnostics))
    if time_diagnostics:
        result.update(green_tau=gtau, screened_tau=tuple(wc_tau),
            screened_zero=tuple(wc_zero), screened_beta=tuple(wc_beta), sigma_tau=sigma_tau)
    return result


__all__ = ['MatsubaraGrid', 'matsubara_grid', 'dyson_green_and_density', 'gw_matsubara_step']
