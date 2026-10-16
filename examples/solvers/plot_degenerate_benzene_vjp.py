"""Plot the completed native benzene subspace VJP and independent PySCF FD."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(directory=Path("reproducibility/ad_showcase/A03_degenerate_vjp")):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(directory=Path(directory))
    native_bytes = (args.directory/"benzene_gradscf.json").read_bytes()
    native = json.loads(native_bytes)
    independent = json.loads((args.directory/"benzene_pyscf.json").read_text())
    if not all(item.get("complete") and item.get("passed") for item in (native, independent)):
        raise RuntimeError("Complete passing native and independent molecular reports are required")
    if independent["native_sha256"] != hashlib.sha256(native_bytes).hexdigest():
        raise RuntimeError("The independent comparison belongs to another native report")
    if not all(row["passed"] for row in independent["finite_differences"][-2:]):
        raise RuntimeError("The final two independent FD steps must pass")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3))
    fig.subplots_adjust(left=.09, right=.97, top=.87, bottom=.16, hspace=.5, wspace=.36)
    structure, spectrum, gradients, finite = axes.flat
    positions = np.asarray([position for _, position in native["geometry_angstrom"]])
    for index in range(6):
        pair = positions[[index, (index+1) % 6]]
        structure.plot(pair[:, 0], pair[:, 1], color=".35", lw=1.7)
        pair = positions[[index, index+6]]
        structure.plot(pair[:, 0], pair[:, 1], color=".65", lw=1.1)
    structure.scatter(positions[:6, 0], positions[:6, 1], s=110, c="#333333", label="C", zorder=3)
    structure.scatter(positions[6:, 0], positions[6:, 1], s=65, c="white", edgecolors=".45", label="H", zorder=3)
    structure.set_aspect("equal")
    structure.set(xlim=(-3.2, 3.2), ylim=(-3.4, 2.8), xticks=[], yticks=[],
                  title="a  Native RHF reference · regular benzene")
    structure.text(.02, .02, f"{native['nao']} Cartesian AOs\n{native['nocc']} occupied × {native['nvir']} virtual = {native['dimension']} singles",
                   transform=structure.transAxes, fontsize=9, va="bottom")
    structure.legend(frameon=False, loc="upper right", fontsize=9)
    for spine in structure.spines.values():
        spine.set_visible(False)

    energies = np.asarray(native["low_roots_hartree"])
    other = np.asarray(independent["low_roots_hartree"])[:len(energies)]
    roots = np.arange(1, len(energies)+1)
    lo, hi = native["cluster"]["start"], native["cluster"]["stop"]
    spectrum.axvspan(lo+.5, hi+.5, color="#F0E442", alpha=.25, label="Complete bright doublet")
    spectrum.plot(roots, energies, "o-", color="#0072B2", ms=5, lw=.8, label="GradSCF native")
    spectrum.plot(roots[:len(other)], other, "s", color="#D55E00", mfc="none", ms=6, label="Independent PySCF")
    spectrum.set(title="b  Selected isolated doublet", xlabel="Sorted root index", ylabel="Excitation energy / Eh")
    spectrum.text(.03, .97, f"Internal splitting: {native['cluster']['splitting_hartree']:.2e} Eh\n"
                  f"Minimum exterior gap: {native['exterior_gap_hartree']:.5f} Eh",
                  transform=spectrum.transAxes, fontsize=8.5, va="top")
    spectrum.legend(frameon=False, loc="lower right", fontsize=8)

    gradient = np.asarray(native["vjp"])
    last_fd = np.asarray(independent["finite_differences"][-1]["vjp"])
    x = np.arange(2)
    gradients.bar(x-.16, gradient, width=.3, color="#0072B2", alpha=.85, label="Native subspace VJP")
    gradients.bar(x+.16, last_fd, width=.3, facecolor="none", edgecolor="#D55E00", hatch="//",
                  label="Independent projector FD")
    gradients.axhline(0, color=".4", lw=.6)
    gradients.set(xticks=x, xticklabels=[r"$\cos 2\phi$ source", r"$\sin 2\phi$ source"],
                  ylabel=r"$\partial L/\partial\theta_a$ / $e^2a_0^2 E_h^{-1}$",
                  title=r"c  Nonzero response of $L=T_{xx}-T_{yy}+0.5T_{xy}$")
    gradients.legend(frameon=False, fontsize=8, loc="best")

    checks = independent["finite_differences"]
    finite.loglog([row["step_hartree"] for row in checks], [row["max_vjp_error"] for row in checks],
                   "o-", color="#D55E00", ms=6, lw=1.4)
    finite.invert_xaxis()
    finite.set_xticks([row["step_hartree"] for row in checks],
                      [f"{row['step_hartree']:.1e}" for row in checks])
    finite.set(title="d  Independent three-step FD convergence", xlabel="Source step / Eh",
               ylabel=r"Maximum VJP discrepancy / $e^2a_0^2 E_h^{-1}$")
    finite.text(.03, .06, f"JVP–VJP duality error: {native['jvp_vjp_error']:.2e}\n"
                f"SCF energy difference: {independent['errors']['scf_energy_hartree']:.2e} Eh\n"
                f"Cluster energy difference: {independent['errors']['cluster_energy_hartree']:.2e} Eh",
                transform=finite.transAxes, fontsize=8.5, va="bottom")
    for axis in (spectrum, gradients, finite):
        axis.grid(axis="y", color=".9", lw=.6)
    fig.suptitle("Benzene RHF/CIS · complete degenerate-subspace VJP · Cartesian 6-31G*\n"
                 "GradSCF native SCF/integrals and shared implicit response; independent PySCF SCF/projector differences",
                 fontsize=12.5)
    fig.text(.09, .09, r"$T=D^T P_{\rm doublet}D$; both members are retained through the symmetry-breaking source."+"\n"
             r"$V_a=(SW_a+W_aS)/2$ is an inversion-even atomic AO source. Orbitals/integrals are fixed during the response."+"\n"
             "This validates the fixed-reference projected CIS model. Relaxed HF/geometry response and adjoint residual telemetry are not included.",
             fontsize=8.2, va="top", linespacing=1.4)
    for extension in ("png", "pdf"):
        fig.savefig(args.directory/f"benzene_subspace_vjp.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("Saved benzene_subspace_vjp.png/.pdf from matching completed native and independent reports")


if __name__ == "__main__":
    run()
