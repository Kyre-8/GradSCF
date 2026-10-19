"""Eager, moment-anchored causal matrix realizations.

The finite ansatz F(z)=sum_l R_l/(z-p_l), R_l >= 0, has a positive
matrix spectral measure. A scaled Gram factor and Hermitian least squares
construct a controlled approximation to the original samples and first moment.
Rank truncation and the free Hermitian completion are explicit diagnostics;
finite data do not define a unique continuation. Check independent holdout,
particle number and forward-grid stability. No AD rule is supplied.
"""
from dataclasses import dataclass
import numpy as np


def _hermitian(a):
    return (a+a.conj().swapaxes(-1, -2))/2


def _causal_samples(z, values):
    z, values = np.asarray(z, complex), np.asarray(values, complex)
    if (z.ndim != 1 or not len(z) or values.ndim != 3
            or values.shape[0] != len(z) or values.shape[1] != values.shape[2]):
        raise ValueError('Expected z(nw) and matrix values(nw,n,n) shapes')
    if not np.isfinite(z).all() or not np.isfinite(values).all() or np.any(z.imag <= 0):
        raise ValueError('Use finite samples in the open upper complex half-plane')
    return z, values


def causal_gram(z, values, *, moment=None):
    """Return the matrix Nevanlinna Gram for a proper causal self-energy.

    Blocks are P_ij=-(Sigma_i-Sigma_j^dagger)/(z_i-conj(z_j)), with
    frequency-major orbital ordering. A supplied first 1/z moment adds the
    FIRST block row/column: P_00=M1, P_i0=Sigma_i, P_0j=Sigma_j^dagger.
    A positive matrix spectral measure requires this matrix to be PSD.
    The function performs no eigenvalue clipping or causal projection;
    negative eigenvalues therefore remain available for diagnosis. Only
    roundoff anti-Hermitian error in the assembled identity is averaged.

    Matrix Pick positivity is described by Fei, Yeh, Zgid and Gull,
    Phys. Rev. B 104, 165111 (2021), Sec. II.B, arXiv:2107.00788.
    Here the kernel uses the anti-Herglotz sign of a proper self-energy.
    """
    z, values = _causal_samples(z, values)
    count, norb = values.shape[:2]
    blocks = -(values[:, None]-values[None].conj().swapaxes(-1, -2))/(
        z[:, None]-z[None].conj())[:, :, None, None]
    gram = blocks.transpose(0, 2, 1, 3).reshape(count*norb, count*norb)
    if moment is not None:
        moment = np.asarray(moment, complex)
        if moment.shape != (norb, norb) or not np.isfinite(moment).all():
            raise ValueError('The first moment must be a finite matching matrix shape')
        if not np.allclose(moment, moment.conj().T, atol=1e-12, rtol=1e-10):
            raise ValueError('The first moment must be Hermitian')
        augmented = np.empty(((count+1)*norb, (count+1)*norb), complex)
        augmented[:norb, :norb] = moment
        augmented[norb:, norb:] = gram
        augmented[norb:, :norb] = values.reshape(count*norb, norb)
        augmented[:norb, norb:] = augmented[norb:, :norb].conj().T
        gram = augmented
    return _hermitian(gram)


@dataclass(frozen=True)
class PoleFit:
    poles: object
    residues: object
    relative_error: float
    moment_error: float
    iterations: int
    converged: bool
    diagnostics: dict | None = None


def interpolate_causal(z, values, *, moment, tolerance, rank_tolerance=1e-12):
    """Moment-anchored Hermitian realization, without a supplied pole grid.

    This is a controlled finite-data causal approximation, not a unique
    analytic continuation or the Schur algorithm of Fei et al., arXiv:2107.00788.
    PSD Gram factors give columns [B,X_i] with H X_i=conj(z_i) X_i-B.
    Hermitian least squares completes H, with its unconstrained block zero.
    It gives the exact completion for compatible, untruncated data; after
    numerical rank truncation the original-data errors must meet tolerance.
    Only roundoff-negative Gram modes are allowed. Numerical ranks and all
    actual-data certificates are reported; incompatible data/ranks raise.
    No optimization or differentiation contract is supplied.
    """
    z, values = _causal_samples(z, values)
    moment = np.asarray(moment, complex)
    if (moment.shape != values.shape[1:] or not np.isfinite(moment).all()
            or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1):
        raise ValueError('Use a matching finite moment and positive interpolation/rank tolerances')
    n, count = len(moment), len(z)
    eps, tiny = np.finfo(float).eps, np.finfo(float).tiny
    roundoff = 256*eps*(count+1)*n
    mnorm, fnorm = max(np.linalg.norm(moment), tiny), max(np.linalg.norm(values), tiny)
    if np.linalg.norm(moment-moment.conj().T) > roundoff*mnorm:
        raise ValueError('The causal first moment must be Hermitian')

    def gram_modes(samples, anchor, label):
        gram = causal_gram(z, samples, moment=anchor)
        scale = np.repeat(np.r_[1., abs(z)], samples.shape[-1])
        gram = scale[:, None]*gram*scale[None]
        eigenvalue, vector = np.linalg.eigh(gram)
        largest = max(float(np.max(abs(eigenvalue))), tiny)
        negative_limit = 64*eps*len(gram)*largest
        if eigenvalue.min() < -negative_limit:
            raise ValueError(f'{label} causal Gram is not PSD: min={eigenvalue.min():.6e}, '
                             f'roundoff limit={negative_limit:.6e}')
        return gram, eigenvalue, vector, negative_limit

    original, original_eigenvalues, _, original_negative_limit = gram_modes(values, moment, 'Original')
    me, mu = np.linalg.eigh(_hermitian(moment))
    if me.min() < -64*eps*n*max(float(np.max(abs(me))), tiny):
        raise ValueError('The causal first moment must be positive semidefinite')
    support = me > rank_tolerance*max(float(me.max()), tiny)
    report = dict(method='moment-anchored Hermitian realization', rank_tolerance=rank_tolerance,
        moment_rank=int(support.sum()), original_gram_min_eigenvalue=float(original_eigenvalues.min()),
        original_gram_negative_roundoff_limit=original_negative_limit,
        column_scaling='[1,abs(z_1),...,abs(z_m)] after moment-support whitening',
        completion='zero on orthogonal complement of range(X)', roundoff_relative_tolerance=roundoff)
    def reject(message):
        error = ArithmeticError(message)
        error.diagnostics = report
        raise error
    if not support.any():
        if np.linalg.norm(values) != 0 or np.linalg.norm(moment) != 0:
            raise ValueError('Nonzero data or moment cannot be discarded as zero moment support')
        report.update(retained_rank=0, free_complement_dimension=0, interpolation_relative_error=0.)
        return PoleFit(np.empty(0), np.zeros((0, n, n), complex), 0., 0., 0, True, report)
    root = mu[:, support]*np.sqrt(me[support])
    inverse = (mu[:, support]/np.sqrt(me[support])).conj().T
    white = inverse@values@inverse.conj().T
    support_error = float(np.linalg.norm(root@white@root.conj().T-values)/fnorm)
    if support_error > roundoff:
        raise ValueError(f'Moment support/rank is incompatible with original data: {support_error:.6e}')
    gram, eigenvalue, vector, negative_limit = gram_modes(white, np.eye(root.shape[1]), 'Whitened')
    threshold = rank_tolerance*max(float(eigenvalue.max()), tiny)
    keep = eigenvalue > threshold
    feature = np.sqrt(eigenvalue[keep])[:, None]*vector[:, keep].conj().T
    r = root.shape[1]
    b, x = feature[:, :r], feature[:, r:]
    scales = np.repeat(abs(z), r)
    t = x*np.repeat(z.conj(), r)[None]-np.tile(b, (1, count))*scales[None]
    # A complete left basis retains range/null cross blocks when rows > columns.
    u, singular, vh = np.linalg.svd(x, full_matrices=x.shape[0] > x.shape[1])
    ur = u[:, :len(singular)]
    xthreshold = rank_tolerance*max(float(singular.max(initial=0.)), tiny)
    selected = singular > xthreshold
    xp = ((vh[selected].conj().T/singular[selected])@ur[:, selected].conj().T)
    q = ur[:, selected]@ur[:, selected].conj().T
    constraint_scale = max(np.linalg.norm(x*np.repeat(z.conj(), r)[None])
        +np.linalg.norm(np.tile(b, (1, count))*scales[None]), tiny)
    null_error = float(np.linalg.norm(t-t@xp@x)/constraint_scale)
    xt = x.conj().T@t
    hermitian_error = float(np.linalg.norm(xt-xt.conj().T)
                            /max(np.linalg.norm(x)*constraint_scale, tiny))
    report.update(retained_rank=int(keep.sum()), gram_rank_threshold=threshold,
        gram_negative_roundoff_limit=negative_limit, moment_support_relative_error=support_error,
        discarded_gram_relative_norm=float(np.linalg.norm(eigenvalue[~keep])/max(np.linalg.norm(eigenvalue), tiny)),
        x_rank=int(selected.sum()), x_rank_threshold=xthreshold,
        x_condition=float(singular[selected].max()/singular[selected].min()) if selected.any() else np.inf,
        free_complement_dimension=int(keep.sum()-selected.sum()),
        null_constraint_relative_error=null_error, hermitian_constraint_relative_error=hermitian_error)
    h0 = t@xp
    h = h0+h0.conj().T-q@h0.conj().T@q
    anti = float(np.linalg.norm(h-h.conj().T)/max(np.linalg.norm(h), float(abs(z).max()), tiny))
    report.update(raw_hamiltonian_antihermitian_relative_error=anti,
        raw_hx_relative_error=float(np.linalg.norm(h@x-t)/constraint_scale))
    if null_error > tolerance or hermitian_error > roundoff:
        reject('Incompatible null/Hermitian constraints at the stated rank')
    # Solve HXX^dagger+XX^daggerH=TX^dagger+XT^dagger in the left SVD basis.
    # The double-null block is free and set to zero; all other blocks are fixed.
    s = np.zeros(len(u))
    s[:len(singular)] = np.where(selected, singular, 0.)
    m = np.zeros((len(u), len(u)), complex)
    m[:, :len(singular)] = u.conj().T@t@vh.conj().T
    denominator = s[:, None]**2+s[None]**2
    numerator = m*s[None]+m.conj().T*s[:, None]
    completed = np.divide(numerator, denominator, out=np.zeros_like(m), where=denominator > 0)
    h = u@completed@u.conj().T
    anti = float(np.linalg.norm(h-h.conj().T)/max(np.linalg.norm(h), float(abs(z).max()), tiny))
    report.update(hermitian_completion='minimum Frobenius norm of H X - T',
                  hamiltonian_antihermitian_relative_error=anti)
    if anti > roundoff:
        reject('Hermitian completion exceeds roundoff at the stated rank')
    h = _hermitian(h)
    hx_error = float(np.linalg.norm(h@x-t)/constraint_scale)
    xu, tu = x/scales[None], t/scales[None]
    unscaled_scale = max(np.linalg.norm(xu*np.repeat(z.conj(), r)[None])
                        +np.linalg.norm(np.tile(b, (1, count))), tiny)
    original_hx_error = float(np.linalg.norm(h@xu-tu)/unscaled_scale)
    poles, modes = np.linalg.eigh(h)
    coupling = modes.conj().T@b@root.conj().T
    residues = np.einsum('li,lj->lij', coupling.conj(), coupling)
    predicted = np.einsum('wl,lij->wij', 1/(z[:, None]-poles), residues)
    error = float(np.linalg.norm(predicted-values)/fnorm)
    moment_error = float(np.linalg.norm(residues.sum(axis=0)-moment)/mnorm)
    report.update(hx_relative_error=hx_error, unscaled_hx_relative_error=original_hx_error,
        interpolation_relative_error=error, moment_relative_error=moment_error, pole_count=len(poles))
    if (not np.isfinite(error+moment_error+hx_error+original_hx_error)
            or max(error, hx_error, original_hx_error) > tolerance or moment_error > 1e-9):
        reject('Original interpolation/moment/constraint certificate failed')
    return PoleFit(poles, residues, error, moment_error, 0, True, report)


def evaluate_poles(fit, z):
    """Evaluate the fitted proper matrix function at arbitrary complex z."""
    z = np.asarray(z)
    return np.einsum('...l,lij->...ij', 1/(z[..., None]-fit.poles), fit.residues)


def auxiliary_spectrum(fock, fit, *, mu=0.):
    """Dyson poles (absolute energies) and per-spin electron spectral weights.

    The augmented Hermitian matrix reproduces the resolvent with the supplied
    dynamical self-energy. Its weights sum to the number of physical orbitals;
    satellite poles are retained. fit poles are relative to mu, fock absolute.
    """
    fock = np.asarray(fock)
    if fock.shape != fit.residues.shape[1:] or not np.allclose(fock, fock.conj().T, atol=1e-11):
        raise ValueError('Fock must be Hermitian in the fitted orbital frame')
    if not len(fit.poles):
        return np.linalg.eigvalsh(_hermitian(fock)), np.ones(len(fock))
    couplings, bath = [], []
    for pole, residue in zip(fit.poles, fit.residues):
        value, vector = np.linalg.eigh(_hermitian(residue))
        keep = value > max(value.max(), 1e-30)*1e-13
        couplings.append(vector[:, keep]*np.sqrt(np.maximum(value[keep], 0.)))
        bath.extend([pole+mu]*int(keep.sum()))
    coupling = np.concatenate(couplings, axis=1)
    matrix = np.block([[fock, coupling], [coupling.conj().T, np.diag(bath)]])
    energy, vector = np.linalg.eigh(matrix)
    return energy, np.sum(abs(vector[:len(fock)])**2, axis=0)
