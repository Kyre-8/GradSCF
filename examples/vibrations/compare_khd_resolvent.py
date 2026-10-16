"""V07 independent two-mode oscillator-basis resolvent oracle (SciPy only).

Build the excited harmonic Hamiltonian in the ground oscillator basis, then
solve its two complex resolvents directly. This does not use Gaussian overlaps
to construct the oracle, nor diagonalize/reimplement an electronic solver.
The shared Gaussian/KHD result is only the comparison target. No CLI.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.linalg import solve

jax.config.update("jax_enable_x64", True)
from gradscf_tools.khd_raman import linear_dipole_vertices, khd_tensor

OUTPUT = Path("reproducibility/ad_showcase/V07_formaldehyde_khd")


def direct_resolvent(size, wi, wf, rotation, shift, dipole, derivatives, gap, laser, gamma):
    identity = np.eye(size)
    coordinates, momenta_squared, squares = [], [], []
    for mode, omega in enumerate(wi):
        # Compute Q^2 with padding before projection: the top retained diagonal
        # contains the path through |size>, which a truncated Q@Q would lose.
        lowering = np.diag(np.sqrt(np.arange(1, size+2)), 1)
        q = (lowering+lowering.T)/np.sqrt(2*omega)
        p = 1j*np.sqrt(omega/2)*(lowering.T-lowering)
        lift = (lambda x: np.kron(x, identity)) if mode == 0 else (lambda x: np.kron(identity, x))
        coordinates.append(lift(q[:size, :size]))
        squares.append(lift((q@q)[:size, :size]))
        momenta_squared.append(lift((p@p).real[:size, :size]))
    curvature = rotation.T@np.diag(wf**2)@rotation
    linear = rotation.T@(wf**2*shift)
    h = .5*sum(momenta_squared)
    h += .5*sum(curvature[k, k]*squares[k] for k in range(2))
    h += curvature[0, 1]*(coordinates[0]@coordinates[1])
    h += sum(linear[k]*coordinates[k] for k in range(2))
    h += np.eye(size**2)*(gap+.5*np.dot(shift, wf**2*shift)-.5*sum(wf))
    mu = [dipole[j]*np.eye(size**2)+sum(derivatives[j, k]*coordinates[k] for k in range(2)) for j in range(3)]
    source = np.column_stack([value[:, 0] for value in mu])
    target = [np.column_stack([value[:, index] for value in mu]) for index in (size, 1)]
    resonant = np.zeros((len(laser), 2, 3, 3), dtype=complex)
    anti = np.zeros_like(resonant)
    max_residual = 0.
    for l, photon in enumerate(laser):
        matrix = h-np.eye(size**2)*(photon+1j*gamma)
        response = solve(matrix, source)
        max_residual = max(max_residual, float(np.linalg.norm(matrix@response-source)/np.linalg.norm(source)))
        for mode in range(2):
            resonant[l, mode] = target[mode].conj().T@response
            matrix = h+np.eye(size**2)*(photon-wi[mode]-1j*gamma)
            response_anti = solve(matrix, source)
            max_residual = max(max_residual, float(np.linalg.norm(matrix@response_anti-source)/np.linalg.norm(source)))
            anti[l, mode] = (target[mode].conj().T@response_anti).T
    return resonant, anti, max_residual


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wi, wf = np.array([.9, 1.3]), np.array([1.1, .7])
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.15, -.21])
    dipole = np.array([.2+.03j, -.3+.02j, .4-.04j])
    derivatives = np.array([[.11, -.08], [.07, .13], [-.05, .04]])*np.exp(.17j)
    gap, gamma, laser = 4., .04, np.array([2.1, 3.7, 4.7])
    states, a, b = linear_dipole_vertices(wi, wf, rotation, shift, dipole, derivatives, max_quanta=30)
    expected = khd_tensor(gap+states@wf, a, b, wi, laser, gamma)
    records = []
    for size in (8, 12, 18, 24):
        r, anti, residual = direct_resolvent(size, wi, wf, rotation, shift, dipole, derivatives, gap, laser, gamma)
        record = dict(basis_per_axis=size, dimension=size**2, relative_resolvent_residual=residual,
            resonant_max_error=float(np.max(abs(r-expected[0]))), antiresonant_max_error=float(np.max(abs(anti-expected[1]))))
        records.append(record)
        print(record, flush=True)
    assert records[-1]["resonant_max_error"] < 2e-9
    assert records[-1]["antiresonant_max_error"] < 2e-11
    assert records[-1]["relative_resolvent_residual"] < 2e-12
    result = dict(passed=True, method="Independent padded ground-HO Hamiltonian and SciPy direct complex resolvents",
        gamma_hartree=gamma, laser_hartree=laser.tolist(), convergence=records, wall_seconds=perf_counter()-started,
        scope="Synthetic two-mode fixed harmonic/linear-dipole model; not an independent HCHO electronic calculation")
    (OUTPUT/"resolvent_reference.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
