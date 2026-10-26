"""Named and explicit basis construction."""
from .data import load_basis_from_snapshot
from .types import BasisTopology, BasisParameters

def prepare_basis(atom, basis, *, unit="Angstrom", charge=0, spin=0, cart=True):
    """Parse a named/raw basis once into static topology and raw JAX parameters.

    Parameters are not sorted or normalized here. Subsequent evaluations use
    the supplied parameter arrays and include their normalization derivatives.
    Nuclear charges are host-static nonnegative int32 values; zero-charge
    centers supplied through ``MoleculeSpec`` are preserved.
    """
    import jax.numpy as jnp
    import numpy as np
    from gradscf.data.molecule import MoleculeSpec, parse_molecule_spec
    from gradscf.integrals.basis import BasisTopology, BasisParameters

    spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(
        atom, unit=unit, charge=charge, spin=spin)
    charge_error = "Basis nuclear charges must be static nonnegative int32 integers, one per atom."
    try:
        charges = np.asarray(spec.charges)
    except (TypeError, ValueError) as exc:
        raise ValueError(charge_error) from exc
    if (charges.shape != (len(spec.symbols),) or charges.dtype.kind not in "iuf"
            or not np.all(np.isfinite(charges)) or np.any(charges < 0)
            or np.any(charges > np.iinfo(np.int32).max)
            or np.any(charges != np.trunc(charges))):
        raise ValueError(charge_error)
    ls, nprim, nctr, exponents, coefficients, centers = [], [], [], [], [], []
    for i, symbol in enumerate(spec.symbols):
        value = basis[symbol] if isinstance(basis, dict) else basis
        blocks = load_basis_from_snapshot(value, symbol) if isinstance(value, str) else value
        for block in blocks:
            l, rows = int(block[0]), block[1:]
            if not rows or not isinstance(rows[0], (list, tuple)):
                raise ValueError("Expected nonrelativistic [l, [exponent, coefficients...], ...] shells.")
            data = jnp.asarray(rows)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Each primitive requires an exponent and contraction coefficients.")
            ls.append(l)
            nprim.append(data.shape[0])
            nctr.append(data.shape[1]-1)
            exponents.append(data[:, 0])
            coefficients.append(data[:, 1:])
            centers.append(jnp.asarray(spec.coords_bohr[i]))
    topology = BasisTopology(tuple(ls), tuple(nprim), tuple(nctr),
                              tuple(int(z) for z in charges), bool(cart))
    parameters = BasisParameters(tuple(exponents), tuple(coefficients),
                                  jnp.stack(centers), jnp.asarray(spec.coords_bohr))
    return topology, parameters
