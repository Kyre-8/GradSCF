"""Plot saved PBE optimization history and initial/final water structures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def run(data_path=Path("reproducibility/ad_showcase/G04_water_pbe_optimization/gradscf.json"), output=None):
    data_path = Path(data_path)
    output = data_path.parent if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(data_path.read_text())
    method = data.get("label", data["xc"])
    response = (f"level-{data['grid_level']} moving grid" if data.get("grid_level") is not None
                else "full electronic response")
    history = data["history"]
    frames = np.asarray([row["coordinates_bohr"] for row in history])
    calls = np.arange(1, len(history)+1)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    axes[0, 0].plot(calls, 1000*(np.asarray([r["energy_hartree"] for r in history])
                                - data["energy_hartree"]), "o-", color="#337a9b")
    axes[0, 0].set(xlabel="Energy/gradient evaluation", ylabel="Energy above final / mEh",
                   title=f"{method} energy along BFGS optimization")
    axes[0, 1].semilogy(calls, [r["max_gradient"] for r in history], "o-", color="#a45d20")
    if "gradient_tolerance" in data:
        axes[0, 1].axhline(data["gradient_tolerance"], color=".5", ls="--", label="acceptance")
    axes[0, 1].set(xlabel="Energy/gradient evaluation", ylabel="Max |gradient| / (Eh/Bohr)",
                   title="Nuclear stationarity")
    if "gradient_tolerance" in data:
        axes[0, 1].legend(frameon=False)
    lengths = np.linalg.norm(frames[:, 1:] - frames[:, :1], axis=2)
    axes[1, 0].plot(calls, lengths[:, 0], "o-", color="#337a9b", label="O–H1")
    axes[1, 0].plot(calls, lengths[:, 1], "x--", color="#a45d20", label="O–H2")
    axes[1, 0].set(xlabel="Energy/gradient evaluation", ylabel="Bond length / Bohr",
                   title="Equivalent O–H bonds remain equivalent")
    axes[1, 0].legend(frameon=False)
    ax = axes[1, 1]
    for coords, color, style, label in ((frames[0], ".65", "--", "initial"),
                                       (np.asarray(data["coordinates_bohr"]), "#337a9b", "-", "optimized")):
        for atom in (1, 2):
            ax.plot(coords[[0, atom], 1], coords[[0, atom], 2], style, color=color,
                    label=label if atom == 1 else None)
        ax.scatter(coords[:, 1], coords[:, 2], s=[90, 40, 40], c=color, zorder=3)
    ax.set(xlabel="y / Bohr", ylabel="z / Bohr", aspect="equal",
           title=f"Final HOH angle: {data['angle_degrees']:.4f}°")
    ax.legend(frameon=False)
    fig.suptitle(f"Water · {method} / Cartesian {data['basis']} · {response} · full AD forces")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"optimization.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    run()


# Generated on jy_gpu_1, 2026-10-15, from saved results:
# reproducibility/ad_showcase/G04_water_pbe_optimization/optimization.png
# reproducibility/ad_showcase/G04_water_pbe_optimization/optimization.pdf
