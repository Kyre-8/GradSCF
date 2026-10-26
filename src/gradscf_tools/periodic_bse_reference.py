"""Independent NumPy finite-model BSE/optical oracle for small examples.

SCF inputs and the original Coulomb boundary are shared. Fourier products,
complete static Lehmann sums, signed-v Dyson contractions and BSE assembly
are independent of the production product builder and screened solver.
Only transition-space matrices are solved; no nG-by-nG dielectric is formed.
This is not an independent SCF or Coulomb-cutoff implementation.
"""
import numpy as np


def matrix(inputs, coefficients, screening_energies, qp_energies, nocc, *, mesh,
           screened=True, max_rank=512, max_transitions=256, max_memory_bytes=512*1024**2):
    """Full supplied occupied/virtual space; all energies are Hartree."""
    from gradscf.integrals.periodic.coulomb import coulomb_kernel
    c, e, qp = map(np.asarray, (coefficients, screening_energies, qp_energies))
    nk, nmo = e.shape
    if c.ndim != 3 or qp.shape != e.shape or not 0 < nocc < nmo:
        raise ValueError('Use matching restricted orbitals and nonempty occupied/virtual spaces')
    nv, nc = nocc, nmo-nocc
    dimension, rank = nk*nv*nc, 2*nk*nv*nc
    if dimension > max_transitions or (screened and rank > max_rank):
        raise ValueError('Independent reference exceeds the bounded transition space')
    coordinates, lattice, kpoints, gvectors = map(np.asarray,
        (inputs.coords, inputs.lattice, inputs.kpoints, inputs.gvectors))
    ngrid, volume = len(coordinates), np.linalg.det(lattice)
    if int(np.prod(mesh)) != ngrid or 80*ngrid*nk*nmo**2+48*rank**2 > max_memory_bytes:
        raise ValueError('Independent reference FFT shape or pair-memory bound failed')
    fractions = kpoints@lattice.T/(2*np.pi)
    table = np.empty((nk, nk), dtype=int)
    for k in range(nk):
        for q in range(nk):
            distance = fractions[k]-fractions[q]-fractions
            matches = np.flatnonzero(np.max(abs(distance-np.rint(distance)), axis=1) < 1e-10)
            if len(matches) != 1:
                raise ValueError('The independent reference requires a closed uniform k/q mesh')
            table[k, q] = matches[0]
    zero = np.flatnonzero(np.linalg.norm(kpoints, axis=1) < 1e-12)
    if len(zero) != 1:
        raise ValueError('Supply one explicit zero q point')
    psi = np.einsum('kgp,kpm->kgm', np.asarray(inputs.ao[:, 0]), c)
    forward = [(i, a) for i in range(nocc) for a in range(nocc, nmo)]
    pairs = np.array(forward+[(a, i) for i, a in forward])
    signs = np.tile(np.r_[np.ones(len(forward)), -np.ones(len(forward))], nk)
    result = np.diag((qp[:, None, nocc:]-qp[:, :nocc, None]).reshape(-1)).astype(complex)
    for q in range(nk):
        blocks = []
        for k, partner in enumerate(table[:, q]):
            density = psi[k].conj()[:, :, None]*psi[partner][:, None, :]
            wrap = kpoints[partner]-kpoints[k]+kpoints[q]
            density = density*np.exp(1j*(coordinates@wrap))[:, None, None]
            blocks.append(np.fft.fftn(density.reshape(*mesh, nmo, nmo), axes=(0, 1, 2))
                          .reshape(ngrid, nmo, nmo)*np.sqrt(volume)/ngrid)
        rho = np.asarray(blocks)
        # Keep the actual signed finite-part head. There is no sqrt(v) here.
        v = np.asarray(coulomb_kernel(gvectors-kpoints[q], boundary=getattr(inputs, 'boundary', None)))
        if q == zero[0]:
            optical = np.concatenate([r[:, :nocc, nocc:].reshape(ngrid, -1) for r in rho], axis=1)
            result += 2/nk*(optical.conj().T@(v[:, None]*optical))
        if screened:
            transitions = np.concatenate([r[:, pairs[:, 0], pairs[:, 1]] for r in rho], axis=1)
            differences = np.concatenate([e[k, pairs[:, 0]]-e[partner, pairs[:, 1]]
                                           for k, partner in enumerate(table[:, q])])
            weights = 2*signs/(nk*differences)
            gram = transitions.conj().T@(v[:, None]*transitions)
            target = np.concatenate([r[:, nocc:, nocc:].reshape(ngrid, -1) for r in rho], axis=1)
            projected = transitions.conj().T@(v[:, None]*target)
            solved = np.linalg.solve(np.eye(rank)-weights[:, None]*gram, weights[:, None]*projected)
        for k, partner in enumerate(table[:, q]):
            left = rho[k, :, :nocc, :nocc].reshape(ngrid, -1)
            right = rho[k, :, nocc:, nocc:].reshape(ngrid, -1)
            direct = left.conj().T@(v[:, None]*right)
            if screened:
                projection = transitions.conj().T@(v[:, None]*left)
                direct += projection.conj().T@solved[:, k*nc**2:(k+1)*nc**2]
            direct = direct.reshape(nv, nv, nc, nc).transpose(0, 2, 1, 3).reshape(nv*nc, nv*nc)
            result[k*nv*nc:(k+1)*nv*nc, partner*nv*nc:(partner+1)*nv*nc] -= direct/nk
    return result


def polarizability(energies, dipoles, omega, *, eta, lattice, dimension):
    """Literal complex resonant/antiresonant sums divided by periodic measure."""
    energy, moments, grid, cell = map(np.asarray, (energies, dipoles, omega, lattice))
    measure = (np.linalg.norm(cell[0]) if dimension == 1 else
               np.linalg.norm(np.cross(cell[0], cell[1])) if dimension == 2 else np.linalg.det(cell))
    values = []
    for frequency in np.atleast_1d(grid):
        tensor = np.zeros((3, 3), dtype=complex)
        for excitation, dipole in zip(energy, moments):
            tensor += np.outer(dipole, dipole.conj())/(excitation-frequency-1j*eta)
            tensor += np.outer(dipole.conj(), dipole)/(excitation+frequency+1j*eta)
        values.append(tensor/measure)
    result = np.asarray(values)
    return result[0] if grid.ndim == 0 else result


__all__ = ['matrix', 'polarizability']
