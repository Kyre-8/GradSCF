"""GPU4PySCF RHF comparison of conventional and trained aniline bases."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import cupy as cp
import gpu4pyscf
import numpy as np
import pyscf
from pyscf import gto, scf


BASES = (
    ("minao", "minimal"),
    ("sto-3g", "minimal"),
    ("3-21g", "split-valence"),
    ("6-31g", "double-zeta"),
    ("6-31g(d)", "polarized double-zeta"),
    ("6-31g(d,p)", "polarized double-zeta"),
    ("def2-svp", "polarized double-zeta"),
    ("cc-pvdz", "correlation-consistent double-zeta"),
    ("6-311g(d,p)", "polarized triple-zeta"),
    ("def2-tzvp", "polarized triple-zeta"),
    ("def2-tzvpp", "polarized triple-zeta"),
    ("cc-pvtz", "correlation-consistent triple-zeta"),
    ("NNAO-szp663-trained", "trained single-zeta plus polarization"),
)


def _warmup():
    mol=gto.M(atom="H 0 0 0; H 0 0 .74",basis="sto-3g",unit="Angstrom",verbose=0)
    mf=scf.RHF(mol).to_gpu();mf.conv_tol=1e-10;mf.kernel()
    if not mf.converged:raise RuntimeError("GPU4PySCF warmup did not converge.")
    cp.cuda.Stream.null.synchronize()
    del mf,mol;gc.collect();cp.get_default_memory_pool().free_all_blocks()


def _make_molecule(summary,name):
    symbols=summary["symbols"];coords=summary["coords_angstrom"]
    if name.startswith("NNAO"):
        labels=[f"{symbol}{i}" for i,symbol in enumerate(symbols)]
        atom=list(zip(labels,coords));basis=dict(zip(labels,summary["final_basis"]))
    else:
        atom=list(zip(symbols,coords));basis=name
    return gto.M(atom=atom,basis=basis,unit="Angstrom",charge=0,spin=0,
                 cart=False,verbose=0,max_memory=64000)


def _run_case(summary,name,level):
    started=time.perf_counter();mol=_make_molecule(summary,name)
    counts=(mol.aoslice_by_atom()[:,3]-mol.aoslice_by_atom()[:,2]).tolist()
    per_element={symbol:sorted({count for s,count in zip(summary["symbols"],counts) if s==symbol})
                 for symbol in sorted(set(summary["symbols"]))}
    cycles=[];mf=scf.RHF(mol).density_fit(auxbasis="cc-pvqz-jkfit").to_gpu()
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=150
    mf.init_guess="minao";mf.chkfile=None
    mf.callback=lambda envs:cycles.append(int(envs.get("cycle",len(cycles))))
    cp.cuda.Stream.null.synchronize();scf_started=time.perf_counter()
    energy=float(mf.kernel());cp.cuda.Stream.null.synchronize()
    scf_seconds=time.perf_counter()-scf_started
    row=dict(basis=name,level=level,nao=int(mol.nao_nr()),hamiltonian_dimension=int(mol.nao_nr()),
             matrix_elements=int(mol.nao_nr())**2,nelectron=int(mol.nelectron),
             occupied_orbitals=int(mol.nelectron//2),virtual_orbitals=int(mol.nao_nr()-mol.nelectron//2),
             ao_per_element=per_element,energy_hartree=energy,converged=bool(mf.converged),
             scf_cycles=int(getattr(mf,"cycles",len(cycles))),scf_seconds=scf_seconds,
             elapsed_seconds=time.perf_counter()-started,status="pass" if mf.converged else "not_converged")
    print(f"{name:22s} AO={row['nao']:3d} E={energy:.12f} cycles={row['scf_cycles']:3d} time={scf_seconds:.2f}s",flush=True)
    del mf,mol;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return row


def compare(summary_path,output_dir):
    summary_path=Path(summary_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    summary=json.loads(summary_path.read_text())
    if summary["molecule"]!="aniline" or summary["basis"]!="szp663_direct":
        raise ValueError("Expected the trained aniline szp663_direct summary.")
    _warmup();rows=[];started=time.perf_counter()
    for name,level in BASES:
        try:row=_run_case(summary,name,level)
        except Exception as error:
            row=dict(basis=name,level=level,status="error",error=f"{type(error).__name__}: {error}")
            print(name,row["error"],flush=True)
            cp.get_default_memory_pool().free_all_blocks()
        rows.append(row);(output_dir/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
    reference=next((r["energy_hartree"] for r in rows if r["basis"]=="cc-pvtz" and r["status"]=="pass"),None)
    if reference is not None:
        for row in rows:
            if row["status"]=="pass":row["above_cc_pvtz_millihartree"]=1000*(row["energy_hartree"]-reference)
    report=dict(molecule="aniline",method="DF-RHF",backend="GPU4PySCF density fitting",cartesian=False,
                charge=0,spin=0,dtype="float64",geometry=summary["geometry_metadata"],
                settings=dict(conv_tol=1e-12,conv_tol_grad=1e-9,max_cycle=150,
                              auxbasis="cc-pvqz-jkfit",init_guess="minao"),
                energy_reference="cc-pVTZ finite-basis RHF, not CBS",
                nnao_note="Single-geometry trained basis; all rows use the same cc-pVQZ-JKFIT approximation as training.",
                source_summary=str(summary_path),source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                python=platform.python_version(),pyscf_version=pyscf.__version__,
                gpu4pyscf_version=gpu4pyscf.__version__,cupy_version=cp.__version__,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu_name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                elapsed_seconds=time.perf_counter()-started,results=rows)
    (output_dir/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    fields=sorted({key for row in rows for key in row if key!="ao_per_element"})+["ao_per_element"]
    with (output_dir/"comparison.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({**row,"ao_per_element":json.dumps(row.get("ao_per_element"))})
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();compare(args.summary,args.output)
