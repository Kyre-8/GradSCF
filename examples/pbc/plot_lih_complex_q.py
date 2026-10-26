"""Plot the completed independent complex-q gradient-FD comparison."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run(input_path, native_path):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path), native=Path(native_path))
    record = json.loads(args.input.read_text())
    if not record.get("complete") or not record.get("passed"):
        raise ValueError("Only a completed passing independent FD comparison may be plotted")
    native = json.loads(args.native.read_text())
    native_hash = hashlib.sha256(args.native.read_bytes()).hexdigest()
    if (record["identity"]["native_sha256"] != native_hash or not native.get("complete")
            or not native.get("passed") or record["direction"] != [[0., 0., 0.], [1., 0., 0.]]):
        raise ValueError("Require the unchanged completed native data and the declared H-x direction")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    expected = np.asarray(record["native_action_real"])+1j*np.asarray(record["native_action_imag"])
    rows = record["steps"]
    smallest_step = min(rows, key=lambda row:row["step_bohr"])
    observed = np.asarray(smallest_step["combined_real"])+1j*np.asarray(smallest_step["combined_imag"])
    errors = np.array([[np.max(abs(np.asarray(row[name+"_real"])
        +1j*np.asarray(row[name+"_imag"])-expected)) for row in rows] for name in ("cosine", "sine", "combined")])
    if not np.isfinite(expected).all() or not np.isfinite(observed).all() or not np.isfinite(errors).all():
        raise ValueError("The completed comparison contains nonfinite data")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(10.5, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.], hspace=.28, wspace=.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])]
    coordinate = np.arange(6)
    for axis, component, label in zip(axes[:2], (np.real, np.imag), ("Real", "Imaginary")):
        axis.plot(coordinate, component(expected), "o-", color="#246b8e", lw=1., label="GradSCF linear response")
        axis.plot(coordinate, component(observed), "s", color="#c76531", ms=7, mfc="none",
                  label=f"PySCF combined FD, h={smallest_step['step_bohr']:.0e}")
        axis.axhline(0., color=".6", lw=.7)
        axis.set(xticks=coordinate, xticklabels=["Li x", "Li y", "Li z", "H x", "H y", "H z"],
                 ylabel=label+" / Eh bohr⁻²")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    for index, (label, color) in enumerate(zip(("Cosine", "Sine", "Combined"), ("#246b8e", "#c76531", "#7b5d91"))):
        axes[2].bar(np.arange(len(rows))+.24*(index-1), errors[index], width=.22, color=color, label=label)
    axes[2].axhline(record["gates"]["action_hartree_per_bohr2"], color=".3", ls=":", lw=1., label="Declared action gate")
    axes[2].set(xticks=np.arange(len(rows)), xticklabels=[f"{row['step_bohr']:.0e}" for row in rows],
                xlabel="Displacement step / bohr", ylabel="Max absolute error / Eh bohr⁻²", yscale="log")
    axes[2].legend(fontsize=8, loc="best")
    axes[2].grid(axis="y", alpha=.2)
    settings = native["settings"]
    q_label = "1/3" if native["q_evaluated"][2] > 0 else "−1/3"
    fig.suptitle(f"LiH complex force-constant action · q=(0,0,{q_label})\n"
                 f"{settings['xc'].upper()} / {settings['basis']} / {settings['pseudo']} · "
                 f"k={settings['kmesh']} · FFT={settings['mesh']}", fontsize=12)
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.81)
    fig.text(.10, .11, "Direction: H-x only. Full complex six-component action retained; no ASR/conjugacy projection.\n"
             "PySCF uses its own converged primitive/three-cell reference and eight cosine/sine displacement endpoints.\n"
             "This validates one complex-q direction of the declared finite model, not every IFC column or bulk convergence.",
             fontsize=8.5, va="top", linespacing=1.4)
    output = args.input.parent
    for suffix in ("png", "pdf"):
        fig.savefig(output/("lih_complex_q_fd."+suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)
    validation = dict(complete=True, passed=True, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        native_sha256=native_hash,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        errors_hartree_per_bohr2=errors.tolist(), patterns=["cosine", "sine", "combined"],
        steps_bohr=[row["step_bohr"] for row in rows], scope=record["scope"])
    (output/"plot_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print("Plotted completed complex-q finite differences:", errors.tolist(), flush=True)


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41/pyscf_zthird/finite_differences.json"),
        Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41/q_0_0_1.json"))
