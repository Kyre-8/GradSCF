"""Collect executed methane MolGW references and replay native BSE snapshots.

Run the methane example first and supply the reference run directory described
in reproducibility/gw_bse/methane_aug_cc_pvdz/molgw/README.md. This script does
not build or execute MolGW and does not alter production numerical settings.
"""
import argparse
from pathlib import Path
import json,re,time,platform,hashlib
from datetime import datetime, timezone
import jax,numpy as np,yaml
jax.config.update('jax_enable_x64',True)
from gradscf import bse
from gradscf.tools.spectra import HARTREE_TO_EV
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs',type=Path,required=True)
parser.add_argument('--native-output',type=Path,default=Path('outputs/methane_gw_bse'))
parser.add_argument('--archive',type=Path,default=Path('reproducibility/gw_bse/methane_aug_cc_pvdz'))
args=parser.parse_args()
root=args.runs.resolve()
archive=args.archive.resolve()
native=json.loads((archive/'spectrum.json').read_text())
HA_MOLGW=27.21138505
MB_PER_AU=.529177210903**2*100
ordered=lambda x:np.array([x[k] for k in sorted(k for k in x if isinstance(k,int))])
out={'date':datetime.now(timezone.utc).isoformat(),'molgw_revision':'b831818d7a845c36f295d036dc8ceef59daf9991','hardware':platform.platform(),'hostname':platform.node(),'dtype':'float64','backend':'CPU','molgw_ha_ev':HA_MOLGW,'gradscf_ha_ev':HARTREE_TO_EV,'methods':{},'driver_control':{}}
out['native_implementation_branch']=native.get('implementation_branch','legacy: not recorded')
out['native_source_python_sha256']=native.get('source_python_sha256','legacy: not recorded')
out['native_json_sha256']=hashlib.sha256((archive/'spectrum.json').read_bytes()).hexdigest()
out['comparison_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for m,orig in native['methods'].items():
 t=time.perf_counter()
 with np.load(args.native_output/f'{m}_reference.npz') as z:
  ref=bse.BSEReference(**dict(z),nocc=5,screening_occupied=range(5),screening_virtual=range(5,61))
 response=bse.BSE(ref,occupied=range(5),virtual=range(5,34),tda=False,solver='dense',nroots=145,max_dense=320).run()
 assert response.converged.all()
 # Verify that reloading the numerical snapshot reproduces the displayed plot.
 np.testing.assert_allclose(response.e,np.array(orig['excitation_ev'])/HARTREE_TO_EV,atol=2e-12,rtol=0)
 out['methods'][m]={}
 for mode,folder in [('default_cutoff',root/m),('matched_cutoff',root/'full_rank'/m)]:
  path=folder/'bse.yaml' if (folder/'bse.yaml').exists() else folder/'molgw.yaml'
  y=yaml.safe_load(path.read_text())['optical spectrum']['excitations']
  energies=ordered(y['energies'])/HA_MOLGW; f=ordered(y['oscillator strengths'])
  cross=np.loadtxt(folder/('bse_cross_section.dat' if mode=='default_cutoff' else 'photoabsorption_cross_section.dat'))
  omega=cross[:,0]/HA_MOLGW
  own=np.asarray(response.absorption_cross_section(omega,eta=.15/HARTREE_TO_EV,unit='Mb'))
  molcross=cross[:,1]*MB_PER_AU
  qp=np.loadtxt(folder/'ENERGY_QP',skiprows=2)[:,1]
  txt=(folder/'gw.out').read_text()
  assert 'This is the end' in txt and 'postscf calculations (if any) will be skipped' not in txt
  hf=float(re.findall(r'SCF Total Energy \(Ha\):\s+([-0-9.]+)',txt)[-1])
  errs={'hf_ha':abs(hf-native['scf_energy_ha']),
        'qp_max_ha':float(max(abs(qp[:34]-np.array(orig['qp_energy_ha'])[:34]))),
        'bse_max_ha':float(max(abs(energies-np.asarray(response.e)))),
        'bse_max_ev':float(max(abs(energies-np.asarray(response.e)))*HARTREE_TO_EV),
        'oscillator_strength_max':float(max(abs(f-np.asarray(response.oscillator_strength())))),
        'static_polarizability_max_au':float(np.max(abs(np.array(y['static polarizability']).reshape(3,3)-np.asarray(response.polarizability())))),
        'cross_section_max_mb':float(max(abs(own-molcross)))}
  # Match the numerical scope of the earlier molecular oracle, not new loose tolerances.
  if mode=='matched_cutoff':
   assert errs['qp_max_ha']<2e-6 and errs['bse_max_ha']<3e-6
   assert errs['oscillator_strength_max']<2e-5 and errs['static_polarizability_max_au']<3e-4
   assert errs['cross_section_max_mb']/MB_PER_AU<2e-4
  record={'errors':errs,'hf_ha':hf,'qp_energy_ha':qp.tolist(),
    'excitation_ev':(energies*HARTREE_TO_EV).tolist(),'oscillator_strength':f.tolist(),
    'spectrum_energy_ev':(omega*HARTREE_TO_EV).tolist(),'cross_section_mb':molcross.tolist(),
    'cross_section_difference_mb':(own-molcross).tolist(),
    'first_bright_ev':float(energies[np.flatnonzero(f>1e-5)[0]]*HARTREE_TO_EV),
    'ri_rank':108 if mode=='default_cutoff' else 109,
    'molgw_total_time_s':re.findall(r'Total time[^\n]*',txt)}
  if m!='g0w0':
   prev=np.loadtxt(folder/'ENERGY_QP_40',skiprows=2)[:,1]
   record['outer_extra_step_max_ha']=float(max(abs(qp-prev)))
   assert record['outer_extra_step_max_ha']<1e-10
  if mode=='matched_cutoff':
   record['molgw_timing_gw']=yaml.safe_load((folder/'gw.yaml').read_text())['run']['timing']
   record['molgw_timing_bse']=yaml.safe_load((folder/'molgw.yaml').read_text())['run']['timing']
  out['methods'][m][mode]=record
  print(m,mode,errs,flush=True)
 out['methods'][m]['native_bse_replay_seconds']=time.perf_counter()-t
p=root/'g0w0'
a=yaml.safe_load((p/'bse_stock.yaml').read_text())['optical spectrum']['excitations'];b=yaml.safe_load((p/'bse_noop.yaml').read_text())['optical spectrum']['excitations']
for k in ('energies','oscillator strengths'):
 out['driver_control'][k+'_max_difference']=float(max(abs(ordered(a[k])-ordered(b[k]))))
out['driver_control']['cross_section_max_difference']=float(np.max(abs(np.loadtxt(p/'bse_stock_cross.dat')-np.loadtxt(p/'bse_noop_cross.dat'))))
assert max(out['driver_control'].values())==0
(archive/'molgw_comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
