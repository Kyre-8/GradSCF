"""Pinned Phonopy C-kernel check for the actual matched LiH dipole tensors.

No GradSCF import. The independent raw IFC interpolation check and this
dipole-kernel check have different scopes; neither redoes electronic SCF.
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
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive, get_supercell


def run(input_path):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path))
    began = perf_counter()
    versions = dict(phonopy=phonopy.__version__, spglib=spglib.__version__, symfc=symfc.__version__)
    if tuple(versions.values()) != ("2.44.0", "2.5.0", "1.5.0"):
        raise RuntimeError("Use the recorded pinned private-kernel environment")
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    source, arrays_path = args.input/"gradscf.json", args.input/"material.npz"
    native = json.loads(source.read_text())
    if not native.get("complete") or not native.get("passed") or digest(arrays_path) != native["arrays_sha256"]:
        raise ValueError("Require complete, checksummed native material data")
    lattice, positions = np.asarray(native["lattice_bohr"]), np.asarray(native["positions_bohr"])
    atoms = PhonopyAtoms(symbols=["Li", "H"], cell=lattice, positions=positions, masses=native["masses_amu"])
    supercell = get_supercell(atoms, np.eye(3, dtype=int))
    primitive = get_primitive(supercell, np.eye(3))
    np.testing.assert_allclose(primitive.cell, lattice, atol=1e-12, rtol=0)
    np.testing.assert_allclose(primitive.positions, positions, atol=1e-12, rtol=0)
    parameters = dict(born=np.asarray(native["born_charges"]), dielectric=np.asarray(native["epsilon_infinity"]),
        factor=1., G_cutoff=2., Lambda=native["smearing_inverse_bohr"]/(2*np.pi))
    oracle = DynamicalMatrixGL(supercell, primitive, np.zeros((2, 2, 3, 3)), nac_params=parameters, use_openmp=False)
    reciprocal = np.linalg.inv(lattice).T
    integers = np.array(list(itertools.product(range(-native["cutoff"], native["cutoff"]+1), repeat=3)))
    oracle._G_list = np.ascontiguousarray(integers@reciprocal, dtype=float)
    oracle._dd_q0 = np.zeros((2, 3, 3), dtype=complex)
    def matrix(q, direction=None):
        cartesian = np.ascontiguousarray(q@reciprocal, dtype=float)
        direction = None if direction is None else np.ascontiguousarray(direction, dtype=float)
        atomic = oracle._get_c_recip_dipole_dipole(cartesian, direction).reshape(6, 6)
        phase = np.repeat(np.exp(2j*np.pi*(positions@cartesian)), 3)
        return phase[:, None]*atomic*phase.conj()[None, :]
    arrays = np.load(arrays_path)
    values = np.array([matrix(q) for q in arrays["long_range_check_qpoints"]])
    errors = dict(non_gamma=float(np.max(abs(values-arrays["long_range_check_matrices"]))),
        gamma_short_circuit=float(np.max(abs(matrix(np.zeros(3))-arrays["long_range_gamma"]))),
        gamma_directional=float(np.max(abs(matrix(np.zeros(3), arrays["long_range_gamma_direction"])
                                           -arrays["long_range_directional_gamma"]))))
    report = dict(complete=True, passed=bool(max(errors.values()) <= 1e-11), errors_hartree_per_bohr2=errors,
        gate_hartree_per_bohr2=1e-11, versions=versions, native_json_sha256=digest(source),
        native_arrays_sha256=digest(arrays_path), script_sha256=digest(Path(__file__)),
        elapsed_seconds=perf_counter()-began,
        scope="Actual matched LiH reciprocal dipole model versus independent Phonopy C kernel. Raw IFC interpolation is independently checked in the P03 capsule; electronic SCF is not repeated here.")
    (args.input/"phonopy_dipole.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Material Phonopy dipole kernel:", report["passed"], errors, flush=True)
    if not report["passed"]:
        raise RuntimeError("Independent material dipole kernel comparison failed")


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P05_lih_polar_bands"))
