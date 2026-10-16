"""Plot saved water convergence, window sensitivity and solver measurements."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from water_forward_convergence import OUTPUT


plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})
COLORS = {"g": "#2465a3", "w": "#d47a1f", "optical": "#5b8c48"}


def save(figure, name):
    figure.savefig(OUTPUT / f"{name}.png", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def convergence():
    quadrature = json.loads((OUTPUT / "quadrature_comparison.json").read_text())
    windows = json.loads((OUTPUT / "window_comparison.json").read_text())["rows"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    nodes = np.array([50, 100, 200])
    for axis, key, title, unit in zip(axes[0], ("qp_energy_hartree", "qp_weight"),
        ("QP energy quadrature error", "QP residue quadrature error"), ("Hartree", "dimensionless")):
        axis.loglog(nodes, [quadrature["errors"][str(n)][key] for n in nodes],
                    "o-", color="#2465a3", label="Compared with 400 nodes")
        axis.axhline(quadrature["gates"][key], color="0.45", ls="--", label="Declared target")
        axis.set(xlabel="Imaginary-axis nodes", ylabel=f"Maximum error / {unit}", title=title)
        axis.set_xticks(nodes, nodes)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=.18)
        axis.legend(fontsize=9)
    for window in ("g", "w", "optical"):
        rows = [row for row in windows if row["window"] == window and row["count"] < 14
                and row["status"] == "converged_local_roots"]
        count = [row["count"] for row in rows]
        if window != "optical":
            axes[1, 0].semilogy(count, [row["qp_energy_error_hartree"] for row in rows],
                                "o-", color=COLORS[window], label=f"{window.upper()} window")
        for label, style in (("tda", "-"), ("full", "--")):
            errors = [row["optical"][label]["matched_energy_error_hartree"] for row in rows]
            axes[1, 1].semilogy(count, errors, style, marker="o", color=COLORS[window],
                                label=f"{window.upper()}: {label.upper()}")
            rejected = [i for i, row in enumerate(rows) if not row["optical"][label]["all_six_tracked"]]
            if rejected:
                axes[1, 1].plot(np.asarray(count)[rejected], np.asarray(errors)[rejected],
                                "x", color="black", ms=10, mew=1.5)
    for axis, title in zip(axes[1], ("QP window sensitivity", "Tracked optical-state sensitivity")):
        axis.axhline(1e-6, color="0.45", ls=":")
        axis.set(xlabel="Retained virtual orbitals", ylabel="Maximum energy change / Hartree",
                 title=title, xticks=[4, 8, 12, 14])
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle("Water / Cartesian 6-31G*: one fixed 19-orbital reference", fontsize=14)
    all_tracked = all(row["optical"][label]["all_six_tracked"] for row in windows
                      if row["status"] == "converged_local_roots" for label in ("tda", "full"))
    tracking_note = ("All six optical states pass the declared tracking criteria." if all_tracked else
                     "Black crosses mark points where not all six states can be tracked.")
    figure.supxlabel("14 virtuals is the reused zero-error endpoint. Optical-only windows keep QPs identical.\n"
                      + tracking_note, fontsize=9)
    save(figure, "convergence")


def representations():
    rows = json.loads((OUTPUT / "representation_comparison.json").read_text())["rows"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    order = ["dense_direct", "dense_gmres", "davidson_direct", "davidson_gmres"]
    labels = ["Dense / direct", "Dense / GMRES", "Davidson / direct", "Davidson / GMRES"]
    metrics = [("cold_seconds", 1., "Cold compile + solve / s"),
               ("warm_median_seconds", 1., "Warm solve median / s"),
               ("process_rss_kib", 1 / 1024, "Whole-process peak RSS / MiB")]
    for axis, (key, scale, title) in zip(axes, metrics):
        for label, offset, color in (("tda", -.19, "#2465a3"), ("full", .19, "#d47a1f")):
            values = [next(row[key] for row in rows if row["name"] == f"{label}_{name}") * scale
                      for name in order]
            axis.bar(np.arange(4) + offset, values, width=.36, label=label.upper(), color=color)
        axis.set(title=title, xticks=np.arange(4), xticklabels=labels)
        axis.tick_params(axis="x", labelrotation=30, labelsize=9)
        axis.grid(axis="y", alpha=.18)
        if key != "process_rss_kib":
            axis.set_yscale("log")
    axes[0].legend()
    figure.suptitle("Same six states and fixed GW result: solver representation measurements", fontsize=14)
    figure.supxlabel("Fresh CPU processes; Davidson max_space = 80; three warm calls per setting. Shared-node timings are indicative.\n"
                      "RSS includes loading, compilation and independent residual probes.", fontsize=9)
    save(figure, "representations")


def window_spectra():
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout="constrained")
    palette = {4: "#b4493b", 8: "#d49d35", 12: "#43858e", 14: "#202d44"}
    for column, window in enumerate(("g", "w", "optical")):
        for row, label in enumerate(("tda", "full")):
            axis = axes[row, column]
            for count in (4, 8, 12, 14):
                folder = OUTPUT / ("nw200" if count == 14 else f"{window}_virtual{count}")
                if not (folder / f"{label}.npz").exists():
                    continue
                with np.load(folder / f"{label}.npz") as result:
                    axis.plot(result["grid_ev"], result["cross_section"], color=palette[count],
                              lw=1.2, alpha=.9, label=str(count))
            axis.set_title(f"{label.upper()}: vary {window.upper()} virtuals")
            axis.grid(alpha=.16)
            axis.set_xlim(0, 50)
            if column == 0:
                axis.set_ylabel(r"Cross section / bohr$^2$")
            if row == 1:
                axis.set_xlabel("Photon energy / eV")
    axes[0, 0].legend(title="Virtual orbitals", fontsize=9)
    figure.suptitle("Independent window sensitivity at fixed 200-node G0W0", fontsize=14)
    figure.supxlabel("Absolute cross sections; imposed HWHM = 0.01 Hartree. The 14-virtual curve is reused from the full-window endpoint.", fontsize=9)
    save(figure, "window_spectra")


if __name__ == "__main__":
    convergence()
    representations()
    window_spectra()

# Saved-data outputs: convergence.png/pdf, representations.png/pdf,
# window_spectra.png/pdf under reproducibility/ad_showcase/GW_BSE_forward_gates.
# Figures were rendered on jy_gpu_1 and visually checked after retrieval.
