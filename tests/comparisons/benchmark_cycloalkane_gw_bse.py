"""Serial CPU cycloalkane timing; use one fresh process per size and method.

GradSCF repeats share one converged RHF source, but use new GW/BSE objects.
MolGW repeats use fresh directories and the documented comparison executable
(RI cutoff 1e-10, optional QP screening adapter). No production defaults change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)
from gradscf import bse, gto, gw, scf

HA_TO_EV = 27.21138505
METHODS = ("g0w0", "evgw0", "evgw")


def _timed(fn, ready):
    start = time.perf_counter()
    value = fn()
    jax.block_until_ready(ready(value))
    return value, time.perf_counter() - start


def _gradscf_case(atom, *, method, nw, repeats):
    mf, scf_seconds = _timed(
        lambda: scf.RHF(gto.M(atom=atom, basis="sto-3g"), conv_tol=1e-10)
        .density_fit("weigend").run(), lambda x: x.mo_energy)
    if not mf.converged:
        raise ArithmeticError("GradSCF HF did not converge")
    nmo, nocc = len(mf.mo_energy), int(np.count_nonzero(mf.mo_occ))
    nroots = nocc*(nmo-nocc)
    samples = []
    for repeat in range(repeats):
        obj, gw_seconds = _timed(lambda: gw.GW(
            mf, method=method, nw=nw, eta=1e-5, max_cycle=40,
            conv_tol=1e-7, damp=.3 if method != "g0w0" else 0.,
        ).run(), lambda x: x.mo_energy)
        if not obj.converged:
            raise ArithmeticError(f"GradSCF {method} did not converge")
        response, bse_seconds = _timed(lambda: bse.BSE(
            obj, nroots=nroots, tda=False, singlet=True, solver="dense",
            max_dense=nroots, conv_tol=1e-7,
        ).run(), lambda x: x.e)
        if not np.all(response.converged):
            raise ArithmeticError("GradSCF BSE did not converge")
        qp = np.asarray(obj.mo_energy)
        excitation = np.asarray(response.e)
        if samples:
            np.testing.assert_allclose(qp, samples[0]["qp_energy_ha"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(excitation, samples[0]["excitation_ha"], atol=1e-10, rtol=0)
        samples.append(dict(gw_seconds=gw_seconds, bse_seconds=bse_seconds,
            qp_energy_ha=qp.tolist(), excitation_ha=excitation.tolist(),
            qp_residual_max_ha=float(np.max(np.abs(obj.result.qp_residual))),
            bse_residual_max_ha=float(np.max(response.result.residual_norms))))
        print(method, repeat, gw_seconds, bse_seconds, flush=True)
    nmo, nocc = len(mf.mo_energy), int(np.count_nonzero(mf.mo_occ))
    return dict(nmo=nmo, nocc=nocc, ntransition=nocc*(nmo-nocc),
        naux=int(mf._scf_inputs.df_factors.shape[0]), scf_seconds=scf_seconds,
        scf_energy_ha=float(mf.e_tot), gw_converged=True, bse_converged=True,
        samples=samples, warm_median={key:float(np.median([s[key] for s in samples[1:]]))
                                    for key in ("gw_seconds", "bse_seconds")})


def _molgw_input(atom, basis_dir, *, method, nmo, stage):
    postscf = "BSE" if stage == "bse" else {
        "g0w0":"G0W0", "evgw0":"GnW0", "evgw":"evGW"}[method]
    return f"""&molgw
 scf='HF'
 basis_path='{basis_dir}/'
 basis='STO-3G'
 auxil_basis='weigend'
 gaussian_type='CART'
 tolscf=1.0e-10
 nscf=100
 frozencore='no'
 ncoreg=0
 ncorew=0
 nvirtualg={nmo + 1}
 nvirtualw={nmo + 1}
 selfenergy_state_min=1
 selfenergy_state_max={nmo}
 nomega_sigma=4001
 step_sigma=0.0005
 eta=0.00001
 postscf='{postscf}'
 nstep_gw={40 if method != 'g0w0' and stage == 'gw' else 1}
 nexcitation=0
 toldav=1.0e-7
 nstep_dav=100
 tda='no'
 triplet='no'
 print_w='no'
 print_bigrestart='yes'
 print_restart='yes'
 read_restart='{'yes' if stage == 'bse' else 'no'}'
 length_unit='angstrom'
 natom={len(atom.split(';'))}
/
{chr(10).join(x.strip() for x in atom.split(';'))}
"""


def _run_molgw(executable, folder, text, *, stage, method):
    (folder / f"{stage}.in").write_text(text)
    env = dict(os.environ)
    env.pop("MOLGW_COMPARE_QP_SCREENING", None)
    env.pop("MOLGW_COMPARE_OPTICAL_LAST", None)
    # Match the actual W provenance in GradSCF's BSEReference.
    if stage == "bse" and method == "evgw":
        env["MOLGW_COMPARE_QP_SCREENING"] = "yes"
    start = time.perf_counter()
    with (folder / f"{stage}.out").open("w") as stream:
        subprocess.run([str(executable.resolve()), f"{stage}.in"], cwd=folder, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
    wall = time.perf_counter() - start
    output = (folder / f"{stage}.out").read_text()
    if "This is the end" not in output or "postscf calculations (if any) will be skipped" in output:
        raise ArithmeticError(f"MolGW {stage} did not finish successfully")
    if "Davidson diago not converged" in output:
        raise ArithmeticError("MolGW BSE did not converge")
    raw = (folder / "molgw.yaml").read_text()
    (folder / f"{stage}.yaml").write_text(raw)
    return wall, yaml.safe_load(raw), output


def _molgw_case(atom, args, method, n):
    samples = []
    nmo = 7*n  # STO-3G: five functions per carbon, one per hydrogen.
    for repeat in range(args.repeats):
        folder = args.output.parent / "molgw_runs" / f"n{n}_{method}_{repeat}"
        folder.mkdir(parents=True, exist_ok=False)
        gw_wall, gw_data, gw_output = _run_molgw(args.molgw, folder,
            _molgw_input(atom, args.basis_dir, method=method, nmo=nmo,
                         stage="gw"), stage="gw", method=method)
        qp = np.loadtxt(folder / "ENERGY_QP", skiprows=2)[:,1]
        if not np.all(np.isfinite(qp)) or len(qp) != nmo:
            raise ArithmeticError("MolGW did not calculate all QP targets")
        delta = None
        if method != "g0w0":
            histories = [v for k, v in gw_data.items()
                         if re.fullmatch(r"g\d+w\d+ energies", k)]
            previous_data = histories[-2]["spin channel 1"]
            previous = np.array([previous_data[k] for k in sorted(
                k for k in previous_data if isinstance(k, int))])/HA_TO_EV
            delta = float(np.max(np.abs(qp-previous)))
            if delta > 1e-7:
                raise ArithmeticError(f"MolGW outer iteration not converged: {delta}")
        bse_wall, bse_data, bse_output = _run_molgw(args.molgw, folder,
            _molgw_input(atom, args.basis_dir, method=method, nmo=nmo,
                         stage="bse"), stage="bse", method=method)
        energies = bse_data["optical spectrum"]["excitations"]["energies"]
        excitation = np.array([energies[k] for k in sorted(k for k in energies if isinstance(k,int))])/HA_TO_EV
        if len(excitation) != 12*n*n or not np.all(np.isfinite(excitation)):
            raise ArithmeticError("Incomplete MolGW BSE roots")
        samples.append(dict(gw_seconds=float(gw_data["run"]["timing"]["postscf"]),
            bse_seconds=float(bse_data["run"]["timing"]["postscf"]),
            gw_process_seconds=gw_wall, bse_process_seconds=bse_wall,
            scf_energy_ha=float(re.findall(r"SCF Total Energy \(Ha\):\s+([-0-9.]+)", gw_output)[-1]),
            ri_rank=int(re.findall(r"load unbalance \(Max - Min\):\s+(\d+)",gw_output)[0]),
            qp_energy_ha=qp.tolist(), excitation_ha=excitation.tolist(),
            outer_last_step_max_ha=delta))
        print("molgw", method, repeat, samples[-1]["gw_seconds"], samples[-1]["bse_seconds"],flush=True)
    return dict(samples=samples, median={key:float(np.median([s[key] for s in samples]))
                                       for key in ("gw_seconds", "bse_seconds")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molgw", type=Path)
    parser.add_argument("--basis-dir", type=Path, default=Path("/private/tmp/molgw-reference-20260930/basis"))
    parser.add_argument("--engine", choices=("gradscf","molgw"), required=True)
    parser.add_argument("--geometries", type=Path, default=Path("reproducibility/gw_bse/cycloalkane_scaling/geometries.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--nw", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or (args.engine == "molgw" and args.molgw is None):
        parser.error("Need at least two repeats and a MolGW executable for MolGW runs")
    case = next(x for x in json.loads(args.geometries.read_text()) if x["n"] == args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(platform=platform.platform(), python=platform.python_version(), jax=jax.__version__,
        backend=jax.default_backend(), device=str(jax.devices()[0]), float64=True,
        threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},
        basis="sto-3g", auxbasis="weigend", nw=args.nw, nroots=12*args.n**2, repeats=args.repeats,
        engine=args.engine, method=args.method, geometry=case)
    report["result"] = (_gradscf_case(case["atom"], method=args.method, nw=args.nw,
        repeats=args.repeats) if args.engine == "gradscf" else
        _molgw_case(case["atom"], args, args.method, args.n))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
