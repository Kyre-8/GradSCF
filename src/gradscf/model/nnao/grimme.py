"""Direct neural contractions on the official qavg-vSZPs primitive/ECP pool.

Averaged coefficients are initialization data, never an additive baseline in
bind(). The inconsistent adaptive file at the pinned upstream commit is excluded.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
from importlib.resources import files
import re
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.data.molecule import parse_molecule_spec,atomic_number
from gradscf.integrals.basis import BasisTopology,BasisParameters
from gradscf.integrals.basis.ecp import AtomicECP
from .basis import supported_elements,DirectBasis


@lru_cache(maxsize=1)
def grimme_templates():
    root=files('gradscf.model.nnao').joinpath('data/qvszps')
    lines=root.joinpath('qavg-vszps_basis_ORCA').read_text().splitlines()
    bases={};symbol=None;i=0
    while i<len(lines):
        fields=lines[i].split();i+=1
        if not fields:continue
        if fields[0].lower()=='newgto':
            symbol=fields[1];bases[symbol]=[]
        elif fields[0].lower()=='end':symbol=None
        elif symbol is not None and fields[0].upper() in 'SPDFGH' and len(fields)==2:
            l='SPDFGH'.index(fields[0].upper());count=int(fields[1]);rows=[]
            for line in lines[i:i+count]:
                row=line.split();rows.append([float(row[1]),float(row[2])])
            i+=count;bases[symbol].append([l,*rows])
    ecps={};z=None;core=None;terms=[];channel=None
    def store():
        if z is not None:ecps[z]=AtomicECP(core,tuple(terms))
    for line in root.joinpath('q-vszp_ecp').read_text().splitlines():
        fields=line.split()
        if not fields or fields[0]=='*':continue
        if len(fields)==1 and fields[0].isdigit():
            store();z=int(fields[0]);core=None;terms=[];channel=None
        elif fields[0]=='ncore':core=int(re.search(r'ncore\s*=\s*(\d+)',line).group(1))
        elif len(fields)==1 and fields[0][0].lower() in 'spdfgh':
            channel='spdfgh'.index(fields[0][0].lower()) if '-' in fields[0] else -1
        elif len(fields)==3 and z is not None:
            terms.append((channel,int(fields[1]),float(fields[2]),float(fields[0])))
        else:raise ValueError(f'Unrecognized ECP record: {line}')
    store()
    return {s:(bases[s],ecps.get(atomic_number(s))) for s in supported_elements()}



def prepare_grimme_basis(atom,*,unit='Angstrom',charge=0,spin=0,cart=False):
    spec=parse_molecule_spec(atom,unit=unit,charge=charge,spin=spin)
    if not np.array_equal(np.asarray(spec.charges),[atomic_number(s) for s in spec.symbols]):
        raise ValueError('Grimme templates require physical nuclear charges; core removal is applied once internally.')
    shells=[];owners=[];potentials=[];effective=[]
    for i,s in enumerate(spec.symbols):
        if s not in grimme_templates():raise ValueError(f'No main-group q-vSZPs template for {s}.')
        basis,ecp=grimme_templates()[s];potentials.append(ecp)
        effective.append(atomic_number(s)-(ecp.ncore if ecp else 0))
        for block in sorted(basis,key=lambda block:block[0]):shells.append(block);owners.append(i)
    ls=tuple(b[0] for b in shells);aa=tuple(jnp.asarray([r[0] for r in b[1:]],dtype=jnp.float64) for b in shells)
    cs=tuple(jnp.asarray([[r[1]] for r in b[1:]],dtype=jnp.float64) for b in shells)
    cs=tuple(c/jnp.linalg.norm(c) for c in cs)
    coords=jnp.asarray(spec.coords_bohr,dtype=jnp.float64)
    top=BasisTopology(ls,tuple(len(a) for a in aa),(1,)*len(ls),tuple(effective),bool(cart))
    parameters=BasisParameters(aa,cs,coords[jnp.asarray(owners)],coords)
    roles=tuple('polarization' if l==2 or (spec.symbols[a] in ('H','He') and l==1) else 'valence' for a,l in zip(owners,ls))
    return DirectBasis(top,parameters,tuple(spec.symbols),tuple(owners),roles,ls,tuple(potentials),spec.charge,spec.spin)


__all__=['prepare_grimme_basis','DirectBasis']
