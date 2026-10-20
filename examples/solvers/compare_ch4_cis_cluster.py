"""Independent relaxed PySCF check and saved-data figure for the CH4 cluster."""
import hashlib
import json
from pathlib import Path

from gradscf_tools.cis_cluster_reference import compare_geometry


OUTPUT = Path("reproducibility/ad_showcase/A03_ch4_cluster_geometry")


def run(directory=OUTPUT):
    directory = Path(directory)
    destination = directory/"pyscf.json"
    if destination.exists():
        raise FileExistsError("Preserve the prior independent record")
    source = directory/"gradscf.json"
    native = json.loads(source.read_text())
    report = compare_geometry(native)
    report["native_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["oracle_source_sha256"] = hashlib.sha256(
        Path("src/gradscf_tools/cis_cluster_reference.py").read_bytes()).hexdigest()
    destination.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise RuntimeError("Independent geometry comparison failed; actual errors are retained")
    print("Independent CH4 geometry tensor response passed:", report["baseline_errors"], flush=True)
    return report


def plot(directory=OUTPUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    path = directory/"gradscf.json"
    native = json.loads(path.read_text())
    oracle = json.loads((directory/"pyscf.json").read_text())
    if not (native["passed"] and oracle["passed"] and oracle["native_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Require passing unchanged native and independent records")
    rows = oracle["finite_differences"]
    labels = [a+b for a in "xyz" for b in "xyz"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].plot(labels, np.asarray(native["tensor_jvp"]).ravel(), "o", label="GradSCF full geometry JVP")
    axes[0].plot(labels, np.asarray(rows[-1]["tensor_derivative"]).ravel(), "s", fillstyle="none",
                 label=f"Independent PySCF FD, h={rows[-1]['step_bohr']:g}")
    axes[0].set(xlabel="Cluster transition tensor component", ylabel="Directional derivative / e² bohr")
    axes[0].legend(fontsize=8)
    steps = [row["step_bohr"] for row in rows]
    axes[1].loglog(steps, [row["tensor_jvp_error"] for row in rows], "o-", label="Full tensor JVP")
    axes[1].loglog(steps, [row["loss_vjp_error"] for row in rows], "s--", label="Weighted-loss VJP")
    axes[1].axhline(oracle["gates"]["directional_derivative"], color=".4", ls=":", label="Declared gate")
    axes[1].set(xlabel="Displacement step / bohr", ylabel="Maximum absolute error / e² bohr")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[["right", "top"]].set_visible(False)
    fig.suptitle("CH₄: fully relaxed geometry response of a bright threefold singlet cluster", fontsize=12)
    fig.text(.07, .02, "RHF / Cartesian 6-31G* · 90 singles · complete-cluster invariant DᵀPD\n"
        "HF, moving AO metric, integrals and dipoles all respond; no individual-root derivative at degeneracy.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.83, wspace=.34)
    for suffix in ("png", "pdf"):
        fig.savefig(directory/f"ch4_cluster_geometry.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

# c20 / PySCF2.13.0: independent RHF energy error1.71e-13 Eh and center
# transition-tensor error1.38e-10 e^2 bohr^2. Rebuilt-HF FD tensor errors are
# 2.27e-6/2.04e-7 e^2 bohr for h=1e-3/3e-4 bohr; both original gates pass.
