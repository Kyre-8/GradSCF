"""Independent Phonopy interpolation of the same raw finite q-mesh IFC data.

No GradSCF import. Electronic force constants are shared input data; this
checks Fourier/image/phase/mass conventions, not independent electronic SCF.
Use the existing pinned Phonopy 2.44 / Spglib 2.5 / Symfc 1.5 oracle environment.
"""
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import phonopy
import spglib
import symfc
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.constants import physical_constants, h, c


def run(input_path):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path))
    started = perf_counter()
    versions = dict(phonopy=phonopy.__version__, spglib=spglib.__version__, symfc=symfc.__version__)
    if tuple(versions.values()) != ("2.44.0", "2.5.0", "1.5.0"):
        raise RuntimeError("Use the recorded pinned oracle environment")
    path, array_path = args.input/"results.json", args.input/"ifc_data.npz"
    native = json.loads(path.read_text())
    fingerprint = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if not native.get("complete") or not native.get("passed") or native.get("asr_projected"):
        raise ValueError("Require complete raw native IFC output, without ASR")
    if fingerprint(array_path) != native["arrays_sha256"]:
        raise ValueError("Native interpolation arrays changed")
    result_path = args.input/"phonopy.json"
    result_path.write_text('{"complete": false, "passed": false}\n')
    arrays = np.load(array_path)
    lattice, positions = arrays["lattice_bohr"], arrays["coordinates_bohr"]
    masses, mesh = arrays["masses_amu"], np.asarray(native["qmesh"])
    atoms = PhonopyAtoms(symbols=["Li", "H"], cell=lattice, positions=positions, masses=masses)
    oracle = Phonopy(atoms, supercell_matrix=np.diag(mesh), primitive_matrix=np.eye(3),
                      is_symmetry=False, hermitianize_dynamical_matrix=False)
    np.testing.assert_allclose(oracle.primitive.cell, lattice, atol=1e-11, rtol=0)
    np.testing.assert_allclose(oracle.primitive.positions, positions, atol=1e-11, rtol=0)
    if oracle.primitive.symbols != ["Li", "H"]:
        raise ValueError("Unexpected primitive atom order")
    tau = positions@np.linalg.inv(lattice)
    owner = np.asarray([{"Li":0, "H":1}[symbol] for symbol in oracle.supercell.symbols])
    shifts = oracle.supercell.positions@np.linalg.inv(lattice)-tau[owner]
    np.testing.assert_allclose(shifts, np.rint(shifts), atol=1e-10, rtol=0)
    shifts = np.rint(shifts).astype(int)
    translations = np.array(list(itertools.product(*[range(n) for n in mesh])))
    # Independent discrete inverse Fourier transform. No native shortest-image
    # list is imported; Phonopy constructs its own shortest vectors below.
    phase = np.exp(-2j*np.pi*(arrays["qpoints"]@translations.T))
    coefficients = np.einsum("qr,qij->rij", phase, arrays["matrices"])/len(translations)
    imaginary = float(np.max(abs(coefficients.imag)))
    if imaginary > 1e-8:
        raise ValueError("Raw q conjugacy failed before constructing real-space IFCs")
    coefficients = coefficients.real.reshape(len(translations), 2, 3, 2, 3)
    full = np.empty((len(owner), len(owner), 3, 3))
    for left in range(len(owner)):
        for right in range(len(owner)):
            index = np.ravel_multi_index(tuple((shifts[right]-shifts[left]) % mesh), tuple(mesh))
            full[left, right] = coefficients[index, owner[left], :, owner[right], :]
    oracle.force_constants = np.ascontiguousarray(full)
    scale = np.sqrt(np.repeat(masses, 3))
    ratio = physical_constants["atomic mass constant"][0]/physical_constants["electron mass"][0]

    def evaluate(q):
        # Phonopy includes atomic positions in its phase. Restore the lattice-
        # translation convention using U=diag(exp(2*pi*i*q.tau)).
        dynamical = oracle.get_dynamical_matrix_at_q(q)
        gauge = np.repeat(np.exp(2j*np.pi*(tau@q)), 3)
        lattice_gauge = gauge[:, None]*dynamical*gauge.conj()[None, :]
        matrix = lattice_gauge*scale[:, None]*scale[None, :]
        eigenvalues = np.linalg.eigvalsh(dynamical)/ratio
        frequencies = np.sign(eigenvalues)*np.sqrt(abs(eigenvalues))
        return matrix, frequencies

    samples = np.array([evaluate(q)[0] for q in arrays["qpoints"]])
    check = np.array([evaluate(q)[0] for q in arrays["check_qpoints"]])
    bands = np.array([evaluate(q)[1] for q in arrays["band_qpoints"]])
    conversion = physical_constants["Hartree energy"][0]/(h*c*100)
    errors = dict(sample_matrix_hartree_per_bohr2=float(np.max(abs(samples-arrays["matrices"]))),
        offgrid_matrix_hartree_per_bohr2=float(np.max(abs(check-arrays["check_matrices"]))),
        band_frequency_hartree=float(np.max(abs(bands-arrays["band_frequencies_hartree"]))),
        raw_ifft_imaginary_hartree_per_bohr2=imaginary)
    passed = max(errors.values()) <= 2e-10
    report = dict(complete=True, passed=bool(passed), versions=versions, errors=errors,
        band_frequency_error_cm1=errors["band_frequency_hartree"]*conversion,
        native_json_sha256=fingerprint(path), native_arrays_sha256=fingerprint(array_path),
        script_sha256=fingerprint(Path(__file__)), elapsed_seconds=perf_counter()-started,
        supercell_atoms=len(owner), is_symmetry=False, hermitianize_dynamical_matrix=False,
        asr_projected=False, force_constant_units="Eh/bohr^2", masses_units="amu",
        phase="Phonopy atomic-position gauge converted by U=exp(2*pi*i*q.tau); unweighted Phi compared.",
        scope="Independent Fourier/shortest-image interpolation and bands from shared raw electronic IFC samples; not independent electronic-structure validation.")
    result_path.write_text(json.dumps(report, indent=2)+"\n")
    print("Phonopy IFC interpolation:", passed, errors, flush=True)
    if not passed:
        raise RuntimeError("Independent Phonopy interpolation gate failed")


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41/ifc_raw"))
