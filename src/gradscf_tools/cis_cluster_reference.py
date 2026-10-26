"""Optional independent PySCF oracle for the small relaxed CIS-cluster example.

No native density, MO coefficients, integrals or operator enter this adapter.
PySCF/SciPy imports are lazy and this module is not used by the native path.
"""
from time import perf_counter
import numpy as np


def evaluate(symbols, coordinates, basis, bounds):
    """Rebuild RHF and the full singlet CIS matrix at one geometry (bohr)."""
    import pyscf
    from pyscf import gto, scf
    from scipy.linalg import eigh

    molecule = gto.M(atom=list(zip(symbols, np.asarray(coordinates))), unit="Bohr",
                     basis=basis, cart=True, verbose=0)
    mf = scf.RHF(molecule)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent PySCF RHF did not converge")
    nocc = int(np.count_nonzero(mf.mo_occ))
    nvir = mf.mo_coeff.shape[1]-nocc
    dimension = nocc*nvir
    start, stop = bounds
    if not 0 <= start < stop <= dimension:
        raise ValueError("Invalid independent cluster bounds")
    td = mf.TDA()
    td.singlet = True
    a, _ = td.get_ab()
    values, vectors = eigh(a.reshape(dimension, dimension))
    gaps = [values[start]-values[start-1]] if start else []
    if stop < dimension:
        gaps.append(values[stop]-values[stop-1])
    if gaps and min(gaps) <= 1e-5:
        raise RuntimeError("An independent displaced cluster lost its exterior gap")
    position = molecule.intor("int1e_r", comp=3)
    probes = -np.sqrt(2)*np.einsum("pi,xpq,qa->iax", mf.mo_coeff[:, :nocc],
        position, mf.mo_coeff[:, nocc:]).reshape(dimension, 3)
    moments = vectors[:, start:stop].T@probes
    return dict(energy_hartree=float(mf.e_tot), excitation_energies_hartree=values.tolist(),
        excitation_energy_sum_hartree=float(values[start:stop].sum()),
        splitting_hartree=float(np.ptp(values[start:stop])), external_gaps_hartree=list(map(float, gaps)),
        transition_tensor=(moments.T@moments).tolist(), nao=molecule.nao, dimension=dimension,
        pyscf=pyscf.__version__, scf_converged=True)


def compare_geometry(native, *, steps=(1e-3, 3e-4)):
    """Compare two fully reconverged directional differences to native JVP/VJP."""
    if not native.get("complete") or not native.get("passed") or native.get("stage") != "response":
        raise ValueError("Require a complete passing native geometry-response record")
    if not native.get("cart"):
        raise ValueError("The independent model requires the same Cartesian basis")
    start = perf_counter()
    symbols, basis = tuple(native["symbols"]), native["basis"]
    coordinates, direction, weights = map(np.asarray,
        (native["coordinates_bohr"], native["direction"], native["tensor_weights"]))
    chosen = native["discovery"]["selected"]
    bounds = (chosen["start"], chosen["stop"])
    center = evaluate(symbols, coordinates, basis, bounds)
    baseline = dict(energy_hartree=abs(center["energy_hartree"]-native["rhf"]["energy_hartree"]),
        excitation_energy_sum_hartree=abs(center["excitation_energy_sum_hartree"]-native["excitation_energy_sum_hartree"]),
        transition_tensor=float(np.max(abs(np.asarray(center["transition_tensor"])-native["transition_tensor"]))))
    gates = dict(energy_hartree=1e-8, excitation_energy_sum_hartree=1e-7,
                 transition_tensor=2e-7, directional_derivative=5e-6)
    rows = []
    for step in steps:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Finite-difference steps must be positive and finite")
        plus = evaluate(symbols, coordinates+step*direction, basis, bounds)
        minus = evaluate(symbols, coordinates-step*direction, basis, bounds)
        difference = (np.asarray(plus["transition_tensor"])-minus["transition_tensor"])/(2*step)
        loss_derivative = float(np.sum(weights*difference))
        tensor_error = float(np.max(abs(difference-native["tensor_jvp"])))
        loss_error = abs(loss_derivative-native["directional_vjp"])
        rows.append(dict(step_bohr=float(step), plus=plus, minus=minus,
            tensor_derivative=difference.tolist(), directional_derivative=loss_derivative,
            tensor_jvp_error=tensor_error, loss_vjp_error=loss_error,
            passed=tensor_error <= gates["directional_derivative"] and loss_error <= gates["directional_derivative"]))
    passed = all(baseline[key] <= gates[key] for key in baseline) and all(row["passed"] for row in rows)
    return dict(complete=True, passed=bool(passed), center=center, baseline_errors=baseline,
        finite_differences=rows, gates=gates, elapsed_seconds=perf_counter()-start,
        scope="Independent RHF/Cartesian-basis/CIS/dipole reconstruction at the center and every displaced geometry. Two directional differences validate the full-tensor JVP and its contraction with the recorded full-coordinate VJP; not a finite-difference check of all 15 components.")


__all__ = ["evaluate", "compare_geometry"]
