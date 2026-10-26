"""Small independent NumPy full-response oracle for arbitrary-k periodic GW.

The SCF and band orbitals are inputs from GradSCF; this is not an independent
SCF implementation. Fourier sums, full Lehmann chi, dense dielectric solves,
exchange and both contour branches below are independent of the GW helpers.
Only small 3D grids (at most 512 auxiliary vectors) are accepted. This oracle
evaluates Dyson residuals at supplied energies, and does not certify a root
branch by importing the native root solver or its reduced screening cache.
"""
import numpy as np


def _factors(inputs, left, right, left_k, right_k, q):
    """Literal Fourier integral, with no FFT or production product builder."""
    coordinates = np.asarray(inputs.coords)
    vectors = np.asarray(inputs.gvectors)-q
    ngrid = len(coordinates)
    norm = np.einsum('gi,gi->g', vectors, vectors)
    coulomb = np.divide(4*np.pi, norm, out=np.zeros_like(norm), where=norm > 1e-24)
    volume = float(inputs.weights[0])*ngrid
    fourier = np.exp(-1j*(vectors@coordinates.T))
    bloch = np.exp(1j*(coordinates@(right_k-left_k)))
    products = left.conj()[:, :, None]*right[:, None, :]*bloch[:, None, None]
    return (np.sqrt(volume*coulomb)[:, None, None]/ngrid
            *(fourier@products.reshape(ngrid, -1)).reshape(ngrid, left.shape[1], right.shape[1]))


def evaluate_query(mf, kpts, orbs, trial_energy, *, nw=32, eta=1e-3):
    """Return independent Sigma_x, Sigma_c and Dyson residuals (Hartree)."""
    from gradscf.integrals.periodic.fft import build_band_inputs
    from gradscf.pbc._kpoint import grid_xc_potential, project_xc_potential
    from gradscf.dft.xc import xc_type
    from gradscf.pbc.momentum import wrap_to_kmesh

    inputs = mf.inputs
    if getattr(inputs, 'boundary', None) is not None or len(inputs.coords) > 512:
        raise ValueError('The independent dense oracle is restricted to small 3D grids')
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(inputs.lattice)).T
    points = wrap_to_kmesh(np.asarray(kpts)@np.linalg.inv(reciprocal))@reciprocal
    source_points = np.asarray(inputs.kpoints)
    source_energy = np.asarray(mf.result.mo_energy_spin[0])
    source_orbitals = np.einsum('kga,kam->kgm', np.asarray(inputs.ao[:, 0]),
                               np.asarray(mf.result.mo_coeff_spin[0]))
    nk, nmo = source_energy.shape
    nq, selected = len(points), np.asarray(orbs)
    nocc = int(np.sum(np.asarray(mf.result.mo_occ_spin[0, 0])))
    occupied = np.arange(nmo) < nocc
    shifted = np.array([wrap_to_kmesh((points+q)@np.linalg.inv(reciprocal))@reciprocal
                        for q in source_points])
    all_points = np.concatenate([points, shifted.reshape(-1, 3)])
    energy, coefficients = mf.get_bands(all_points)
    energy, coefficients = np.asarray(energy), np.asarray(coefficients)
    all_inputs = [build_band_inputs(mf.cell, p, reference=inputs) for p in [points, *shifted]]
    target_orbitals = np.einsum('kga,kam->kgm', np.asarray(all_inputs[0].ao[:, 0]), coefficients[:nq])
    union = np.concatenate([source_energy, energy])
    ef = (np.max(union[:, nocc-1])+np.min(union[:, nocc]))/2
    _, potential = grid_xc_potential(mf.result.density_spin, inputs, xc=mf.xc)
    vxc = np.asarray(project_xc_potential(all_inputs[0].ao, inputs.weights, potential, xc_type(mf.xc))[0])
    vxc_diagonal = np.einsum('kam,kab,kbm->km', coefficients[:nq].conj(), vxc, coefficients[:nq]).real
    roots, raw_weights = np.polynomial.legendre.leggauss(int(nw))
    freqs, weights = .5*(1+roots)/(1-roots), raw_weights/(1-roots)**2
    exchange = np.zeros((nq, len(selected)))
    sigma = np.zeros_like(exchange, dtype=complex)
    identity = np.eye(len(inputs.coords), dtype=complex)
    largest_odd = 0.
    for q, transfer in enumerate(source_points):
        factors, differences, signs = [], [], []
        for k, source in enumerate(source_points):
            # Independent partner search, using the public canonical convention.
            residual = wrap_to_kmesh((source_points-source+transfer)@np.linalg.inv(reciprocal))
            matches = np.flatnonzero(np.max(np.abs(residual), axis=1) < 1e-8)
            if len(matches) != 1:
                raise ValueError('The source mesh must close under every q transfer')
            partner = int(matches[0])
            pair = _factors(inputs, source_orbitals[k], source_orbitals[partner], source,
                            source_points[partner], transfer)
            for m in range(nmo):
                for n in range(nmo):
                    sign = int(occupied[m])-int(occupied[n])
                    if sign:
                        factors.append(pair[:, m, n])
                        differences.append(source_energy[k, m]-source_energy[partner, n])
                        signs.append(sign)
        factors = np.stack(factors, axis=1)
        differences, signs = np.asarray(differences), np.asarray(signs)
        internal_energy = energy[(q+1)*nq:(q+2)*nq]
        internal_orbitals = np.einsum('kga,kam->kgm', np.asarray(all_inputs[q+1].ao[:, 0]),
                                      coefficients[(q+1)*nq:(q+2)*nq])
        external = np.stack([_factors(inputs, internal_orbitals[k], target_orbitals[k],
                             shifted[q, k], points[k], transfer)[:, :, selected] for k in range(nq)], axis=1)
        exchange -= np.sum(np.abs(external[:, :, :nocc])**2, axis=(0, 2))/nk
        flat_external = external.reshape(len(inputs.coords), -1)
        def chi(z, imaginary=True):
            denominator = z+differences+(0. if imaginary else 1j*eta*signs)
            return (factors*(2*signs/(nk*denominator))[None, :])@factors.conj().T
        positive, negative = [], []
        for frequency in freqs:
            for response, z in ((positive, 1j*frequency), (negative, -1j*frequency)):
                action = np.linalg.solve(identity-chi(z), flat_external)-flat_external
                response.append(np.sum(flat_external.conj()*action, axis=0).reshape(nq, nmo, len(selected))/nk)
        positive, negative = np.array(positive), np.array(negative)
        largest_odd = max(largest_odd, float(np.max(np.abs(positive-negative))))
        for k in range(nq):
            for pos, p in enumerate(selected):
                omega = np.asarray(trial_energy)[k, p]
                delta = omega-internal_energy[k]-1j*eta*np.sign(ef-internal_energy[k])
                # Direct two-side formula, independent of even/odd production split.
                sigma[k, pos] -= np.sum(weights[:, None]*(positive[:, k, :, pos]
                    /(delta[None, :]-1j*freqs[:, None])+negative[:, k, :, pos]
                    /(delta[None, :]+1j*freqs[:, None])))/(2*np.pi)
                for m, internal in enumerate(internal_energy[k]):
                    if not (ef < internal < omega or omega < internal < ef):
                        continue
                    pole = omega-internal+1j*eta*np.sign(internal-ef)
                    vector = external[:, k, m, pos]
                    action = np.linalg.solve(identity-chi(pole, imaginary=False), vector)-vector
                    sigma[k, pos] += np.sign(omega-ef)*np.vdot(vector, action)/nk
    ewald = np.zeros_like(exchange)
    if mf.exxdiv == 'ewald':
        ewald -= float(inputs.madelung)*occupied[selected][None, :]
    exchange += ewald
    result = np.asarray(trial_energy)[:, selected]-energy[:nq, selected]-exchange+vxc_diagonal[:, selected]-sigma.real
    full_sigma = np.zeros_like(trial_energy, dtype=complex)
    full_residual = np.zeros_like(trial_energy, dtype=float)
    full_sigma[:, selected], full_residual[:, selected] = sigma, result
    return dict(residual=full_residual, sigma_c=full_sigma, sigma_x=exchange,
        ewald_correction=ewald, ks_energy=energy[:nq], vxc=vxc_diagonal,
        fermi_energy=float(ef), max_two_sided_difference=largest_odd,
        scope='Independent NumPy Fourier/full chi/dense W/Sigma; shared SCF and band orbitals')


__all__ = ['evaluate_query']
