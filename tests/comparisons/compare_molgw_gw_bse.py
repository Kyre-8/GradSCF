"""Opt-in molecular MolGW/GradSCF comparison; never downloads or installs software.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python this_file.py
       /path/to/molgw --workdir /tmp/molgw-comparison --output report.json

Set the executable's runtime library environment before calling this script.
MolGW source: b831818d7a845c36f295d036dc8ceef59daf9991 (3.4).
The default seven-case baseline uses HF/STO-3G, including matched Weigend RI
for the independent virtual-window case. --suite extended adds eight cases
with 6-31G, larger molecules, and water evGW0/evGW. Per-case settings include
the MolGW self-energy grid and an explicit N2 RHF multistart. --keep-going
preserves partial diagnostics and exits nonzero if any case fails.
Atomic units unless explicitly marked.
"""
from pathlib import Path
import argparse
from dataclasses import dataclass
import re
import hashlib
import json
import platform
import subprocess
import time

import jax
import numpy as np
import yaml

jax.config.update('jax_enable_x64', True)
from gradscf import bse, gto, gw, scf

MOLGW_REVISION = 'b831818d7a845c36f295d036dc8ceef59daf9991'
MOLGW_HA_EV = 27.21138505
@dataclass(frozen=True)
class Case:
    name: str
    atom: str
    method: str = 'g0w0'
    gcore: int = 0
    wcore: int = 0
    wend: int | None = None
    basis: str = 'sto-3g'
    auxbasis: str | None = None
    scf_amplitudes: tuple[float, ...] = ()
    sigma_points: int = 40001
    sigma_step: float = .00005


H2 = 'H 0 0 0; H 0 0 .74'
WATER = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
CASES = (
    Case('h2_g0w0',H2), Case('h2_evgw0',H2,method='evgw0'),
    Case('h2_evgw',H2,method='evgw'), Case('water_g0w0',WATER),
    Case('water_w_core',WATER,wcore=1), Case('water_gw_core',WATER,gcore=1,wcore=1),
    Case('water_w_virtual_ri',WATER,wend=6,auxbasis='weigend'),
)
EXTENDED_CASES = (
    Case('h2_631g_g0w0',H2,basis='6-31g'),
    Case('water_631g_g0w0',WATER,basis='6-31g',sigma_points=400001,sigma_step=.000005),
    Case('ammonia_sto3g','N 0 0 .1165; H 0 .9397 -.2718; H .8138 -.4699 -.2718; H -.8138 -.4699 -.2718'),
    Case('methane_sto3g','C 0 0 0; H .629 .629 .629; H .629 -.629 -.629; H -.629 .629 -.629; H -.629 -.629 .629'),
    Case('nitrogen_sto3g','N 0 0 -.55; N 0 0 .55',scf_amplitudes=(.15,)),
    Case('ethylene_sto3g','C -.667 0 0; C .667 0 0; H -1.232 .928 0; H -1.232 -.928 0; H 1.232 .928 0; H 1.232 -.928 0'),
    Case('water_sto3g_evgw0',WATER,method='evgw0'),
    Case('water_sto3g_evgw',WATER,method='evgw'),
)



def execute(executable, folder, name, text):
    (folder / (name+'.in')).write_text(text)
    with (folder / (name+'.out')).open('w') as stream:
        subprocess.run([str(executable),name+'.in'],cwd=folder,check=True,
                       stdout=stream,stderr=subprocess.STDOUT)
    result=yaml.safe_load((folder/'molgw.yaml').read_text())
    (folder/(name+'.yaml')).write_text((folder/'molgw.yaml').read_text())
    return result


def ordered(mapping):
    return np.array([mapping[k] for k in sorted(k for k in mapping if isinstance(k,int))])


def compare_case(executable, folder, case, *, nw=200, sigma_points=None,
                 sigma_step=None, qp_solver='secant'):
    name, atom, method = case.name, case.atom, case.method
    gcore, wcore, wend = case.gcore, case.wcore, case.wend
    sigma_points=case.sigma_points if sigma_points is None else sigma_points
    sigma_step=case.sigma_step if sigma_step is None else sigma_step
    folder.mkdir(parents=True,exist_ok=True)
    # Refuse to silently reuse a cached W or an incompatible SCF restart.
    if any(folder.iterdir()):
        raise FileExistsError(f'Use a fresh comparison directory: {folder}')
    start=time.perf_counter()
    auxbasis = case.auxbasis
    mf=scf.RHF(gto.M(atom=atom,basis=case.basis),conv_tol=1e-12)
    if auxbasis is not None:
        mf.density_fit(auxbasis)
        mf.df_tol = 1e-6  # MolGW's absolute Coulomb-metric eigenvalue cutoff
    scf_attempts=[]
    if case.scf_amplitudes:
        from dataclasses import asdict
        branches=mf.multistart(amplitudes=case.scf_amplitudes,seed=20260923)
        scf_attempts=[asdict(a) for a in branches.attempts]
        mf=branches.selected
        if mf is None:raise ArithmeticError('No converged restricted SCF branch')
    else:
        mf.run()
    if not mf.converged: raise ArithmeticError('GradSCF HF did not converge')
    nmo=len(mf.mo_energy);nocc=int(np.count_nonzero(mf.mo_occ))
    wend=nmo if wend is None else wend
    # MolGW restricts QP targets along with the G core cutoff.
    # The independent W-virtual cutoff uses RI: MolGW's no-RI route also
    # truncates its W product representation, so it is not the same model.
    gend = nmo
    targets=tuple(range(gcore,gend))
    gi=tuple(range(gcore,gend)); oi=tuple(range(wcore,nocc)); va=tuple(range(nocc,wend))
    lines=atom.split(';')
    common=f"""&molgw
 scf='HF'
 basis='{case.basis.upper()}'
 gaussian_type='CART'
 tolscf=1.0e-12
 nscf=100
 frozencore='no'
 ncoreg={gcore}
 ncorew={wcore}
 nvirtualg={gend+1}
 nvirtualw={wend+1}
 selfenergy_state_min={gcore+1}
 selfenergy_state_max={nmo}
 nomega_sigma={sigma_points}
 step_sigma={sigma_step}
 length_unit='angstrom'
 natom={len(lines)}
"""
    if auxbasis is not None: common += f" auxil_basis='{auxbasis}'\n"
    geometry='/\n'+'\n'.join(x.strip() for x in lines)+'\n'
    keyword={'g0w0':'G0W0','evgw0':'GnW0','evgw':'evGW'}[method]
    steps=1 if method == 'g0w0' else 40
    print_w='yes' if auxbasis is None else 'no'
    gw_input=common+f" postscf='{keyword}'\n nstep_gw={steps}\n eta=0.00001\n print_w='{print_w}'\n"+geometry
    upstream=execute(executable,folder,'gw',gw_input)
    if not upstream.get('scf is converged',False):raise ArithmeticError('MolGW SCF did not converge')
    qp_reference=np.loadtxt(folder/'ENERGY_QP',skiprows=2)[:,1]
    partial=dict(name=name,atom=atom,basis=case.basis,method=method,
                 qp_hartree=qp_reference.tolist(),gradscf_hf_hartree=float(mf.e_tot))
    (folder/'comparison.partial.json').write_text(json.dumps(partial,indent=2))
    own=gw.GW(mf,method=method,nw=nw,eta=1e-5,max_cycle=100,conv_tol=1e-10,qp_solver=qp_solver,
              g_orbitals=gi,screening_occupied=oi,screening_virtual=va).run(orbs=targets)
    if not own.converged:
        partial.update(gradscf_qp_hartree=np.asarray(own.mo_energy).tolist(),
                       residuals=np.asarray(own.result.qp_residual).tolist(),
                       converged_mask=np.asarray(own.result.converged_mask).tolist())
        (folder/'comparison.partial.json').write_text(json.dumps(partial,indent=2))
        raise ArithmeticError('GradSCF GW did not converge')
    own_qp=np.asarray(own.mo_energy)
    target=np.asarray(targets)
    matches=re.findall(r'SCF Total Energy \(Ha\):\s+([-+0-9.]+)',(folder/'gw.out').read_text())
    hf_reference=float(matches[-1]) if matches else upstream['scf energy']['total']
    errors={'hf_hartree':abs(float(mf.e_tot)-hf_reference),
            'qp_hartree':float(np.max(np.abs(own_qp[target]-qp_reference[target])))}
    datum=dict(name=name,atom=atom,basis=case.basis,auxbasis=auxbasis,
               hf_hartree=hf_reference,nw=nw,qp_solver=qp_solver,
               scf_amplitudes=case.scf_amplitudes,scf_seed=20260923,scf_attempts=scf_attempts,
               sigma_points=sigma_points,sigma_step=sigma_step,method=method,g_orbitals=gi,
               screening_occupied=oi,screening_virtual=va,qp_targets=targets,
               molgw_input=gw_input,qp_hartree=qp_reference.tolist(),
               gradscf_qp_hartree=own_qp.tolist(),
               gradscf_qp_residual_max=float(np.max(np.abs(own.result.qp_residual))),
               gradscf_qp_weight=np.asarray(own.result.qp_weight)[target].tolist(),
               errors=errors,optical=[])
    # Nonlinear MolGW iterations do not emit the same weight section.
    se=upstream.get('gw selfenergy',{})
    if 'renormalization factor' in se:
        weight=ordered(se['renormalization factor']['spin channel 1'])
        datum['qp_weight']=weight.tolist()
        errors['qp_weight']=float(np.max(np.abs(np.asarray(own.result.qp_weight)[target]-weight)))
    (folder/'comparison.partial.json').write_text(json.dumps(datum,indent=2))
    for tda,singlet in ((True,True),(False,True),(False,False)):
        label=('tda' if tda else 'full')+('_singlet' if singlet else '_triplet')
        text=common+f" postscf='BSE'\n eta=0.01\n tda='{'yes' if tda else 'no'}'\n triplet='{'no' if singlet else 'yes'}'\n read_restart='yes'\n"+geometry
        external=execute(executable,folder,label,text)['optical spectrum']['excitations']
        energies=ordered(external['energies'])/MOLGW_HA_EV
        strengths=ordered(external['oscillator strengths'])
        response=bse.BSE(own,occupied=oi,virtual=va,nroots=len(oi)*len(va),
                         tda=tda,singlet=singlet,solver='dense').run()
        if not response.converged.all():raise ArithmeticError('BSE did not converge')
        err=dict(energy_hartree=float(np.max(np.abs(response.e-energies))),
                 oscillator_strength=float(np.max(np.abs(response.oscillator_strength()-strengths))))
        out=dict(tda=tda,singlet=singlet,energies_hartree=energies.tolist(),
                 oscillator_strengths=strengths.tolist(),errors=err)
        if singlet:
            static=np.array(external['static polarizability']).reshape(3,3)
            err['static_polarizability_au']=float(np.max(np.abs(response.polarizability()-static)))
            # Sample MolGW's actual output grid, including its maximum intensity.
            data=np.loadtxt(folder/'photoabsorption_cross_section.dat')
            indices=np.unique(np.r_[np.linspace(0,len(data)-1,17,dtype=int),np.argmax(data[:,1])])
            frequency=data[indices,0]/MOLGW_HA_EV
            cross=np.asarray(response.absorption_cross_section(frequency,eta=.01))
            err['cross_section_au']=float(np.max(np.abs(cross-data[indices,1])))
            out.update(static_polarizability_au=static.tolist(),omega_hartree=frequency.tolist(),
                       cross_section_au=data[indices,1].tolist())
        datum['optical'].append(out)
        (folder/'comparison.partial.json').write_text(json.dumps(datum,indent=2))
    datum['seconds']=time.perf_counter()-start
    return datum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--source',type=Path,help='Pinned MolGW source; default executable parent')
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=['baseline','extended'],default='baseline')
    parser.add_argument('--case',action='append',choices=[c.name for c in CASES+EXTENDED_CASES])
    parser.add_argument('--nw',type=int,default=200)
    parser.add_argument('--sigma-points',type=int,help='Override per-case frequency grid')
    parser.add_argument('--sigma-step',type=float,help='Override per-case spacing in Ha')
    parser.add_argument('--qp-solver',choices=['secant','newton','hybrid'],default='secant')
    parser.add_argument('--keep-going',action='store_true',help='Record failures, continue, then exit nonzero')
    args=parser.parse_args()
    exe=args.executable.resolve()
    source=(args.source or exe.parent).resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision != MOLGW_REVISION:
        raise ValueError(f'Expected MolGW {MOLGW_REVISION}, got {revision}')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=source,text=True).strip():
        raise ValueError('Reference MolGW tracked sources must be unmodified')
    hashes={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in
            ('src/m_selfenergy_tools.f90','src/m_selfenergy_evaluation.f90',
             'src/m_linear_response.f90','src/m_spectral_function.f90','src/m_spectra.f90')}
    report=dict(molgw_revision=revision,source_sha256=hashes,executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                backend=jax.default_backend(),jax=jax.__version__,platform=platform.platform(),
                python=platform.python_version(),dtype='float64',molgw_ha_ev=MOLGW_HA_EV,
                settings=dict(nw=args.nw,gw_eta_hartree=1e-5,optical_eta_hartree=.01,
                              molgw_sigma_points=args.sigma_points,molgw_sigma_step_hartree=args.sigma_step),
                cases=[],failures=[])
    selected=CASES+EXTENDED_CASES if args.case else (CASES if args.suite=='baseline' else EXTENDED_CASES)
    for case in selected:
        if args.case and case.name not in args.case:continue
        try:
            record=compare_case(exe,args.workdir/case.name,case,nw=args.nw,
                                sigma_points=args.sigma_points,sigma_step=args.sigma_step,
                                qp_solver=args.qp_solver)
            report['cases'].append(record)
            # Preserve the original absolute tolerances.
            limits={'hf_hartree':1e-7,'qp_hartree':2e-6,'qp_weight':5e-5}
            optical_limits={'energy_hartree':3e-6,'oscillator_strength':2e-5,
                            'static_polarizability_au':3e-4,'cross_section_au':2e-4}
            failed=[f'{k}: {v} >= {limits[k]}' for k,v in record['errors'].items() if not np.isfinite(v) or v>=limits[k]]
            failed += [f'{k}: {v} >= {optical_limits[k]}' for o in record['optical'] for k,v in o['errors'].items()
                       if not np.isfinite(v) or v>=optical_limits[k]]
            if failed:report['failures'].append(dict(case=case.name,kind='tolerance',details=failed))
            print(case.name,record['errors'],[x['errors'] for x in record['optical']],flush=True)
        except (ArithmeticError,RuntimeError,ValueError,subprocess.CalledProcessError) as error:
            failure=dict(case=case.name,kind=type(error).__name__,details=str(error))
            partial=args.workdir/case.name/'comparison.partial.json'
            if partial.exists():failure['partial']=json.loads(partial.read_text())
            report['failures'].append(failure)
            print('FAILED',case.name,type(error).__name__,str(error),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if report['failures'] and not args.keep_going:break
    if report['failures']:
        raise SystemExit(f"{len(report['failures'])} comparison(s) failed; see {args.output}")


if __name__=='__main__':main()
