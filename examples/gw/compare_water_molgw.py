"""Independent comparison of saved native water G0W0/BSE and pinned MolGW.

No SCF/GW calculation or root fitting is performed here. The native calculation
must have selected its QP seeds without reading this independent reference.
The reference directory records the fixed MolGW revision, executable hashes,
graphical-root settings and original inputs. All 70 singlet roots are compared.
"""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse
from water_primary_qp import OUTPUT


REFERENCE = Path("reproducibility/ad_showcase/MolGW_reference_build/water_forward/g0w0")
QP_REFERENCE = REFERENCE.parent/"g0w0_grid5e-7"
MOLGW_HARTREE_TO_EV = 27.21138505


def compare(*, nw=200, qp_reference=QP_REFERENCE, optical_reference=REFERENCE, comparison_name="refined"):
    output = OUTPUT/f"nw{nw}"
    native = json.loads((output/"native.json").read_text())
    qp_reference, optical_reference = Path(qp_reference), Path(optical_reference)
    reference = json.loads((qp_reference/"reference.json").read_text())
    optics = json.loads((optical_reference/"reference.json").read_text())
    for source in (reference, optics):
        assert source["complete"] and native["method"] == source["method"] == "g0w0"
        assert native["basis"].lower() == source["basis"].lower()
        for key in ("cart", "nmo", "nocc", "qp_orbitals", "g_orbitals",
                    "screening_occupied", "screening_virtual", "eta_gw_hartree"):
            assert native[key] == source[key], key
        np.testing.assert_allclose(native["coordinates_bohr"], source["coordinates_bohr"], atol=0, rtol=0)
    limits = dict(hf_hartree=1e-7, qp_hartree=2e-6, qp_weight=5e-5)
    optical_limits = dict(energy_hartree=3e-6, oscillator_strength=2e-5,
                          static_polarizability_au=3e-4, cross_section_au=2e-4)
    maximum = lambda first, second: float(np.max(abs(np.asarray(first)-np.asarray(second))))
    report = dict(validation_passed=False, nw=nw, qp_reference_directory=str(qp_reference),
        optical_reference_directory=str(optical_reference),
        qp_reference_step_hartree=reference["sigma_step_hartree"],
        optical_reference_qp_step_hartree=optics["sigma_step_hartree"],
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (output/"native.json", qp_reference/"reference.json", optical_reference/"reference.json", Path(__file__))},
        errors=dict(hf_hartree=abs(native["hf_energy_hartree"]-reference["hf_energy_hartree"]),
                    qp_hartree=maximum(native["qp_energy_hartree"], reference["qp_energy_hartree"]),
                    qp_weight=maximum(native["qp_weight"], reference["qp_weight"])),
        limits=limits, optical_limits=optical_limits, optical=[],
        reference_hartree_to_ev=MOLGW_HARTREE_TO_EV,
        scope="Separate native and MolGW wavefunctions, full G/W/QP and optical spaces, matched Cartesian basis. QP and optical reference directories are recorded separately; finer QP data do not imply recalculated optical data.")
    for own in native["optical"]:
        label = "tda" if own["tda"] else "full"
        external = next(row for row in optics["optical"] if row["tda"] == own["tda"])
        with np.load(output/f"{label}_response.npz") as data:
            arrays = dict(data)
        dipole = arrays.pop("dipole_mo")
        for key in ("singlet", "amplitude_response"):
            arrays[key] = bool(arrays[key])
        result = bse.BSEResult(**arrays)
        space = bse.make_bse_space(native["nmo"], native["nocc"])
        # MolGW's formatted grid is converted using its own energy constant.
        # Evaluate the existing native property routine on those exact points.
        spectrum = np.loadtxt(optical_reference/f"{label}_singlet_cross_section.dat")
        omega = spectrum[:, 0]/MOLGW_HARTREE_TO_EV
        cross = np.asarray(bse.absorption_cross_section(result, dipole, space, omega, eta=.01, unit="au"))
        errors = dict(energy_hartree=maximum(own["excitation_hartree"], external["excitation_hartree"]),
            oscillator_strength=maximum(own["oscillator_strength"], external["oscillator_strength"]),
            static_polarizability_au=maximum(own["static_polarizability_au"],
                                            np.asarray(external["static_polarizability_au"]).reshape(3, 3)),
            cross_section_au=maximum(cross, spectrum[:, 1]))
        report["optical"].append(dict(tda=own["tda"], errors=errors))
        np.savetxt(output/f"{label}_molgw_{comparison_name}_comparison.csv",
                   np.column_stack((spectrum[:, 0], cross, spectrum[:, 1], cross-spectrum[:, 1])),
                   delimiter=",", header="energy_eV,GradSCF_bohr2,MolGW_bohr2,difference_bohr2", comments="")
        print(label, errors, flush=True)
    accepted = all(np.isfinite(value) and value < limits[key] for key, value in report["errors"].items())
    accepted &= all(np.isfinite(value) and value < optical_limits[key]
                    for row in report["optical"] for key, value in row["errors"].items())
    report["validation_passed"] = bool(accepted)
    (output/f"comparison_{comparison_name}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("QP/HF", report["errors"], "passed", accepted, flush=True)
    assert accepted, "The saved native/MolGW comparison failed its original reference gates."
    return report


if __name__ == "__main__":
    compare()

# Independent MolGW refinement (actual executions; fixed eta and energy window):
# QP grid step/Eh     max residue difference
# 5e-6                 1.65073e-4  (failed 5e-5 gate)
# 1e-6                 6.39444e-5  (failed 5e-5 gate)
# 5e-7                 3.52844e-5  (passed original gate)
# Only QP references were refined. The original 70-state TDA/full-BSE optical
# references remain separately identified; they were not recomputed on fine grids.
