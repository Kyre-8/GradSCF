"""Verified Cartesian signed-permutation symmetries for native s/p RHF response.

This deliberately covers only axis permutations/sign changes (at most 48),
not a general point-group search. Geometry and basis are never symmetrized.
"""
from itertools import permutations, product

import numpy as np


def signed_permutation_symmetry(model, coordinates, reference, *, tolerance=1e-8):
    """Return verified ``(T, Q)`` coordinate and reference-MO representations.

    ``model`` is a NativeRHF, with one contraction per s/p shell. Coordinates
    and reference matrices must be real. Operations
    preserve nuclear charges and shell ordering/parameters within each atom.
    Invalid AO or RHF-state invariance raises instead of silently applying an
    approximate symmetry. Coordinates and geometry tolerance are in bohr.
    With ``T e_i = sign e_j``, exact response columns transform as
    ``H[:, j] = sign*T@H[:, i]`` and ``dF[j] = sign*Q@dF[i]@Q.T``.
    """
    topology, parameters = model.topology, model.parameters
    if any(l not in (0, 1) for l in topology.angular_momenta):
        raise ValueError('Symmetry supports only s/p shells.')
    if any(n != 1 for n in topology.contraction_counts):
        raise ValueError('Symmetry requires one contraction per shell.')
    if np.iscomplexobj(coordinates):
        raise ValueError('Symmetry requires real coordinates.')
    coordinates = np.asarray(coordinates, dtype=float)
    charges = np.asarray(topology.nuclear_charges)
    if (coordinates.shape != (len(charges), 3) or not np.all(np.isfinite(coordinates))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Coordinates must be finite (natom, 3); tolerance must be positive.')
    owners = np.asarray(model.owners)
    shells = [np.flatnonzero(owners == atom) for atom in range(len(charges))]
    sizes = np.array([1 if l == 0 else 3 for l in topology.angular_momenta])
    offsets = np.r_[0, np.cumsum(sizes)]
    c = np.asarray(reference.mo_coeff)
    s, f = np.asarray(reference.overlap_matrix), np.asarray(reference.fock_matrix)
    d = np.asarray(reference.density_matrix)
    if any(np.iscomplexobj(a) for a in (c, s, f, d)):
        raise ValueError('Symmetry requires real reference matrices.')
    nao = topology.nao
    if any(a.shape != (nao, nao) or not np.all(np.isfinite(a)) for a in (c, s, f, d)):
        raise ValueError('Reference must have complete finite AO/MO matrices.')
    if not bool(reference.converged):
        raise ValueError('Symmetry requires a converged RHF reference.')
    fmo = c.T @ f @ c
    centered = coordinates-coordinates.mean(axis=0)
    operations = []
    for axes in permutations(range(3)):
        for signs in product((1., -1.), repeat=3):
            rotation = np.eye(3)[list(axes)]*np.asarray(signs)[:, None]
            distances = np.linalg.norm((centered@rotation.T)[:, None]-centered[None], axis=-1)
            targets = np.argmin(distances, axis=1)
            if (np.max(distances[np.arange(len(charges)), targets]) > tolerance
                    or len(set(targets)) != len(charges)
                    or not np.array_equal(charges, charges[targets])):
                continue
            t, p = np.zeros((coordinates.size, coordinates.size)), np.zeros((nao, nao))
            for atom, target in enumerate(targets):
                t[3*target:3*target+3, 3*atom:3*atom+3] = rotation
                if len(shells[atom]) != len(shells[target]):
                    raise ValueError('Symmetry-related atoms have mismatched shell basis.')
                for source_shell, target_shell in zip(shells[atom], shells[target]):
                    a, b = source_shell, target_shell
                    if (topology.angular_momenta[a] != topology.angular_momenta[b]
                            or not np.array_equal(parameters.exponents[a], parameters.exponents[b])
                            or not np.array_equal(parameters.coefficients[a], parameters.coefficients[b])):
                        raise ValueError('Symmetry-related atoms have mismatched shell basis.')
                    p[offsets[b]:offsets[b+1], offsets[a]:offsets[a+1]] = (
                        rotation if sizes[a] == 3 else 1.)
            q = c.T @ s @ p @ c
            for actual, expected in ((p@s@p.T, s), (p@f@p.T, f), (p@d@p.T, d),
                                     (q@q.T, np.eye(nao)), (q@fmo@q.T, fmo)):
                if not np.allclose(actual, expected, atol=tolerance, rtol=0):
                    raise ValueError('AO/MO symmetry invariance failed for the RHF reference.')
            operations.append((t, q))
    return operations


def coordinate_orbits(operations):
    """Return ascending leaders and ``(leader, operation_index, sign)`` per axis.

    Input is the complete operation list from :func:`signed_permutation_symmetry`.
    The first representative for each Cartesian orbit is retained without any
    averaging; callers evaluate only the leaders and transform their columns.
    """
    if not operations:
        raise ValueError('At least the identity symmetry operation is required.')
    mapping = [None]*operations[0][0].shape[0]
    leaders = []
    for i in range(len(mapping)):
        if mapping[i] is not None:
            continue
        leaders.append(i)
        for operation, (t, _) in enumerate(operations):
            j = int(np.argmax(np.abs(t[:, i])))
            if mapping[j] is None:
                mapping[j] = (i, operation, int(t[j, i]))
    return leaders, mapping
