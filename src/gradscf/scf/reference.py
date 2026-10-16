"""SCF reference preparation: AO snapshots and distinct post-HF Hamiltonians."""

from dataclasses import dataclass
import hashlib
import numpy as np
import jax.numpy as jnp
from gradscf.integrals.molecular.ao2mo import transform_integrals, validate_integrals, validate_unrestricted_integrals, unrestricted_frozen_indices, transform_unrestricted_integrals


@dataclass(frozen=True)
class RestrictedReference:
    h1: object
    eri: object
    nocc: int
    nuclear_repulsion: object = 0.0
    mo_energy: object = None


@dataclass(frozen=True)
class UnrestrictedReference:
    """Real orthonormal alpha/beta MOs, occupied first in each spin channel.

    h1=(ha,hb), eri=(gaa,gab,gbb) in chemists' notation; nocc=(na,nb).
    A ROHF reference uses the same spatial orbital frame in both channels.
    """
    h1: object
    eri: object
    nocc: tuple[int, int]
    nuclear_repulsion: object = 0.0
    mo_energy: object = None


def unrestricted_reference_from_source(source):
    if isinstance(source, UnrestrictedReference):
        h, g = validate_unrestricted_integrals(source.h1, source.eri)
        unrestricted_frozen_indices(h[0].shape[0], source.nocc)
        return UnrestrictedReference(h, g, tuple(source.nocc), source.nuclear_repulsion, source.mo_energy)
    from .facade import UKS
    from .roks import ROKS
    if not isinstance(source, (UKS, ROKS)) or str(source.xc).strip().lower() != "hf":
        raise NotImplementedError("Unrestricted post-HF requires a UHF/ROHF HF reference or UnrestrictedReference")
    if not source.converged:
        raise RuntimeError("Run and converge the HF SCF before post-HF")
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before post-HF")
    if isinstance(source, ROKS):
        inputs = source._scf_inputs
        if inputs is None or source.scf_result is None:
            raise RuntimeError("Run the ROHF SCF before post-HF")
        nocc = (inputs.nalpha, inputs.nbeta)
        coeff = (source.mo_coeff, source.mo_coeff)
        expected = (np.arange(source.mo_coeff.shape[1]) < nocc[0]).astype(int)
        expected += (np.arange(source.mo_coeff.shape[1]) < nocc[1]).astype(int)
        hcore, eri, factors = inputs.hcore, inputs.eri, inputs.df_factors
        enuc = inputs.nuclear_repulsion
    else:
        ref = source.reference
        if ref is None:
            raise RuntimeError("Run the UHF SCF before post-HF")
        nocc = (ref.nocc_alpha, ref.nocc_beta)
        coeff = source.mo_coeff
        expected = np.stack([np.arange(coeff.shape[2]) < no for no in nocc])
        hcore, eri, factors = ref.h1e, ref.rep_tensor, ref.df_factors
        enuc = ref.nuclear_repulsion
    if not np.allclose(np.asarray(source.mo_occ), expected, atol=1e-10, rtol=0):
        raise ValueError("Post-HF requires integer occupations, occupied orbitals first in each spin channel")
    if factors is not None:
        kwargs = {"df_factors": factors}
    elif eri is not None:
        kwargs = {"eri" if jnp.ndim(eri) == 4 else "eri_pair_matrix": eri}
    else:
        raise ValueError("HF reference does not supply post-HF two-electron integrals")
    h, g = transform_unrestricted_integrals(hcore, coeff, **kwargs)
    return UnrestrictedReference(h, g, nocc, enuc, source.mo_energy)


def is_unrestricted_source(source):
    from .facade import UKS
    from .roks import ROKS
    return isinstance(source, (UnrestrictedReference, UKS, ROKS))


def reference_from_source(source):
    if isinstance(source, RestrictedReference):
        h, g = validate_integrals(source.h1, source.eri)
        if not isinstance(source.nocc, int) or not 0 <= source.nocc <= h.shape[0]:
            raise ValueError("Invalid restricted nocc")
        return RestrictedReference(
            h, g, source.nocc, source.nuclear_repulsion, source.mo_energy
        )
    from .facade import RKS

    if not isinstance(source, RKS) or str(source.xc).strip().lower() != "hf":
        raise NotImplementedError(
            "Post-HF requires a GradSCF closed-shell HF reference (RKS with xc='hf') or RestrictedReference"
        )
    if not source.converged or source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run and converge the HF SCF before post-HF")
    # Detect stale molecular parameters using the facade's existing guard.
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before post-HF")
    inputs = source._scf_inputs
    nocc = inputs.nelectron // 2
    c = source.mo_coeff
    if c.ndim != 2 or inputs.nelectron % 2:
        raise NotImplementedError("Only real closed-shell HF references are supported")
    expected = np.zeros(c.shape[1])
    expected[:nocc] = 2
    if not np.allclose(np.asarray(source.mo_occ), expected, atol=1e-10, rtol=0):
        raise ValueError(
            "Post-HF requires integer occupations, occupied orbitals first"
        )
    if inputs.df_factors is not None:
        kwargs = {"df_factors": inputs.df_factors}
    elif inputs.eri is not None:
        kwargs = {"eri": inputs.eri}
    else:
        pair = inputs.response_eri_pair_matrix()
        if pair is None:
            raise ValueError(
                "SCF reference does not supply post-HF two-electron integrals"
            )
        kwargs = {"eri_pair_matrix": pair}
    h, g = transform_integrals(inputs.hcore, c, **kwargs)
    return RestrictedReference(h, g, nocc, inputs.nuclear_repulsion, source.mo_energy)


def _array_signature(value):
    array = np.asarray(value)
    return array.shape, array.dtype.str, hashlib.sha256(array.tobytes()).digest()


def reference_state_signature(source):
    """Fingerprint a validated eager post-HF input, including mutable arrays."""
    from .rsh import reject_rsh_response
    reject_rsh_response(source)
    if isinstance(source, UnrestrictedReference):
        return (source.nocc, tuple(_array_signature(a) for a in (*source.h1, *source.eri)),
                _array_signature(source.nuclear_repulsion))
    if all(hasattr(source, name) for name in ("h1", "eri", "nocc")):
        return (
            source.nocc,
            _array_signature(source.h1),
            _array_signature(source.eri),
            _array_signature(source.nuclear_repulsion),
        )
    return (
        source._scf_signature(),
        id(source.scf_result),
        id(source.reference) if source.scf_result is None else None,
        source.converged,
        _array_signature(source.mo_coeff),
        _array_signature(source.mo_occ),
    )


def as_reference(source):
    """Resolve an AO/grid state from an SCF facade or an explicit array state.

    This is an eager source adapter, not the MO Hamiltonian adapter used by
    CI/CC and not a GW/BSE reference conversion. The source owns freshness,
    convergence and spin/representation validation.
    """
    from .rsh import reject_rsh_response
    reject_rsh_response(source)
    convert = getattr(source, 'to_reference', None)
    if callable(convert):
        return convert()
    if hasattr(source, 'molecule'):
        molecule = source.molecule
        if molecule is not None:
            return molecule
        ensure = getattr(source, '_ensure_molecule', None)
        if callable(ensure):
            return ensure()
    if hasattr(source, 'reference'):
        ensure = getattr(source, '_ensure_reference', None)
        if callable(ensure):
            return ensure()
        if source.reference is not None:
            return source.reference
        ensure = getattr(source, '_ensure_molecule', None)
        if callable(ensure):
            return ensure()
        raise RuntimeError('Run ground-state mf.kernel() or mf.run() before requesting a reference.')
    return source
