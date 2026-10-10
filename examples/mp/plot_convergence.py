"""Plot measured MP convergence and compare explicitly identified literature data.

Run he2_convergence.py, compare_he2_convergence.py and ne_f_convergence.py first.
All high-order He curves are native GradSCF; atomic high-order curves are the
external PySCF/NumPy reference. No curve smoothing or fitted energies are used.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
folder = Path("artifacts/mp_convergence")
literature = json.loads((here/"convergence_literature.json").read_text())
he = json.loads((folder/"he2_native.json").read_text())
reference = json.loads((folder/"he2_reference.json").read_text())
atoms = [json.loads((folder/(s+"_reference.json")).read_text()) for s in ("ne","f")]
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "svg.fonttype":"none","savefig.dpi":180})
blue,orange,black = "#2468a2","#b55b16","#252525"
summary = {}

fig,axes = plt.subplots(2,2,figsize=(11,8),layout="constrained")
curves = he["curves"]
r = np.array([d["distance"] for d in curves])
emin = min(d["fci"] for d in curves if d["distance"] < .9)
for key,label,color,style in (("rhf","RHF",blue,"--"),("uhf","UHF",orange,"-."),("fci","FCI",black,"-")):
    axes[0,0].plot(r,[(d[key]-emin)*2625.4996394799 for d in curves],
                   label=label,color=color,linestyle=style,lw=1.8)
axes[0,0].set(xlabel=r"He–He distance / $\AA$",ylabel="Relative energy / kJ mol$^{-1}$",
              title="(a) Reference states and exact curve")
axes[0,0].legend(frameon=False)

for distance,style in ((1.074,"-"),(1.8,"--")):
    data = min(curves,key=lambda d:abs(d["distance"]-distance))
    for method,color in (("rmp",blue),("ump",orange)):
        partial = data["rhf" if method == "rmp" else "uhf"]+np.cumsum(data[method])
        error = np.abs(partial-data["fci"])*1000
        axes[0,1].semilogy(np.arange(2,he["order"]+1),np.maximum(error,1e-10),
            color=color,linestyle=style,lw=1.6,label="%s, %.3f Å" % (method.upper(),data["distance"]))
axes[0,1].set(xlabel="MP order",ylabel="Absolute FCI error / mHartree",
              title="(b) Convergence at two separations")
axes[0,1].legend(frameon=False,fontsize=9)

for method,color in (("rmp",blue),("ump",orange)):
    rows = [d for d in he["extrema"] if d["method"] == method]
    x,y = [d["order"] for d in rows],[d["barrier_kj_mol"] for d in rows]
    axes[1,0].plot(x,y,"o-",color=color,label=method.upper()+" (GradSCF)",lw=1.6,ms=5)
    expected = np.asarray(literature["he2"][method+"_barrier_kj_mol"])
    axes[1,0].scatter(x,expected,s=85,facecolors="none",edgecolors=black,
                      label="Gill–Radom, Table 3" if method == "rmp" else None,zorder=4)
    summary[method+"_max_barrier_difference_kj_mol"] = float(np.max(np.abs(np.asarray(y)-expected)))
exact_barrier = next(d["barrier_kj_mol"] for d in he["extrema"] if d["method"] == "fci")
axes[1,0].axhline(exact_barrier,color=black,linestyle=":",label="FCI barrier")
axes[1,0].set(xlabel="MP order",ylabel="Optimized barrier / kJ mol$^{-1}$",
              title="(c) A small MP4 change is not convergence")
axes[1,0].legend(frameon=False,fontsize=9)
axes[1,1].plot(r,[d["spin_square"] for d in curves],color=orange,lw=1.8)
axes[1,1].axhline(0,color=black,linestyle=":",lw=1,label="Pure singlet")
axes[1,1].set(xlabel=r"He–He distance / $\AA$",ylabel=r"UHF $\langle S^2\rangle$",
              title="(d) UHF spin contamination",ylim=(-.03,1.03))
axes[1,1].legend(frameon=False)
for ax in axes.flat:
    ax.grid(alpha=.15)
fig.suptitle(r"He$_2^{2+}$: high-order MP with native GradSCF"+"\nFixed Gill–Radom minimal basis · RHF / broken-symmetry UHF · CPU float64",fontsize=13)
for suffix in ("png","pdf","svg"):
    fig.savefig(folder/("he2_convergence."+suffix))
plt.close(fig)

fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
for data,color,label in zip(atoms,(blue,orange),("Ne",r"F$^{-}$")):
    coeff = np.asarray(data["reference_corrections"])
    orders = np.arange(2,data["order"]+1)
    axes[0].semilogy(orders,np.abs(coeff)*1000,"o-",color=color,ms=4,label=label+" external MP")
    axes[0].scatter([2,3],np.abs(data["native_corrections"])*1000,color=color,
                    marker="x",s=65,zorder=5,label=label+" GradSCF MP2/MP3")
    error = (data["reference_hf"]+np.cumsum(coeff)-data["fci_energy"])*1000
    axes[1].plot(orders,error,"o-",color=color,ms=3.5,label=label)
    summary[data["system"]] = dict(native_hf_error=abs(data["native_hf"]-data["reference_hf"]),
        native_mp23_error=float(np.max(np.abs(np.asarray(data["native_corrections"])-coeff[:2]))),
        fci=data["fci_energy"],fci_residual=data["fci_residual"],
        literature_fci_error=abs(data["fci_energy"]-literature["atomic_fci"][data["system"]]),
        minimum_correction_order=int(orders[np.argmin(np.abs(coeff))]),
        elapsed_seconds=data["elapsed_seconds"])
    np.testing.assert_allclose(data["fci_energy"],literature["atomic_fci"][data["system"]],atol=1e-6,rtol=0.)
ne_literature = np.asarray(literature["ne"]["corrections_millihartree"])
axes[0].scatter(np.arange(2,26),np.abs(ne_literature),facecolors="none",edgecolors=black,
                s=65,label="Ne: Olsen et al., Table III",zorder=4)
summary["Ne"]["max_literature_coefficient_error"] = float(np.max(np.abs(
    np.asarray(atoms[0]["reference_corrections"])-ne_literature/1000)))
np.testing.assert_allclose(atoms[0]["reference_corrections"],ne_literature/1000,atol=2e-9,rtol=0.)
axes[0].set(xlabel="MP order",ylabel="Absolute order correction / mHartree",
            title="(a) Late Ne divergence; early F$^{-}$ divergence")
axes[0].legend(frameon=False,fontsize=8.5)
axes[1].set_yscale("symlog",linthresh=.1)
axes[1].axhline(0,color=black,linewidth=.8)
axes[1].set(xlabel="MP order",ylabel="Signed partial-sum error / mHartree",
            title="(b) Error against independently solved FCI")
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=.15)
fig.suptitle("Ne / F$^{-}$: complete external high-order benchmark\nSpherical aug-cc-pVDZ · frozen 1s · all virtual orbitals · PySCF symmetry action + NumPy RS",fontsize=12)
for suffix in ("png","pdf","svg"):
    fig.savefig(folder/("ne_f_convergence."+suffix))
plt.close(fig)
(folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
print("Figures:",folder.resolve())
