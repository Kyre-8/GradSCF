"""Pointwise spin-LDA diagnostic, independent of integrals and SCF iterations."""
import argparse
import json
from pathlib import Path
from ten_system_scf_matrix import ROOT
import jax
import jax.numpy as jnp
import jax_xc
import numpy as np
from pyscf.dft import libxc
from gradscf.dft.xc._jax_xc import eval_jax_xc_energy_density_from_unrestricted_density_gradients as evaluate


def check():
    jax.config.update('jax_enable_x64',True)
    rows=[]
    for component in ['lda_x','lda_c_pw','lda_c_vwn']:
        for rho in [.01,.1,1.]:
            for polarization in [0.,.2,.8]:
                spin=jnp.asarray([rho*(1+polarization)/2,rho*(1-polarization)/2])
                zero=jnp.zeros(3)
                f=lambda r:evaluate(component,r[0],r[1],zero,zero)
                value,gradient=jax.value_and_grad(f)(spin)
                raw=getattr(jax_xc,component)(polarized=True)(lambda _:spin,zero)
                if isinstance(raw,(tuple,list)):raw=raw[0]
                raw=float(jnp.asarray(raw))*rho
                exc,vxc,_,_=libxc.eval_xc(component,np.asarray(spin)[:,None],spin=1,deriv=1)
                reference=float(exc[0]*rho);vref=np.asarray(vxc[0])[0]
                rows.append(dict(component=component,rho=rho,polarization=polarization,
                    gradscf=float(value),raw_jax_xc=raw,pyscf=reference,
                    energy_density_error=abs(float(value)-reference),raw_energy_density_error=abs(raw-reference),
                    potential_error=float(np.max(np.abs(np.asarray(gradient)-vref)))))
    return dict(jax_xc_version=jax_xc.__version__,rows=rows,
        max_errors={c:max(r['energy_density_error'] for r in rows if r['component']==c) for c in ['lda_x','lda_c_pw','lda_c_vwn']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();s=check();a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s['max_errors'],indent=2))
