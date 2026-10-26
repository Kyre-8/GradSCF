"""Per-atom shell templates and differentiable contraction assembly."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.data.molecule import parse_molecule_spec, MoleculeSpec, atomic_number
from gradscf.integrals.basis import BasisTopology, BasisParameters
from .initialization import _templates, _direct_templates, _direct_slot




def supported_elements():
    return tuple(_templates())


@dataclass(frozen=True)
class NeuralBasis:
    """Static shell mapping; bind outputs [atom, (valence_s,valence_p), tangent].

    Raw coefficients remain in the GradSCF input convention. Integral plans
    perform the physical Gaussian normalization. Padding atoms are not shells.
    """
    topology: BasisTopology
    parameters: BasisParameters
    symbols: tuple[str,...]
    shell_atoms: tuple[int,...]
    roles: tuple[str,...]
    slots: tuple[int,...]
    tangents: tuple
    update_scale: float = 1.

    def bind(self, outputs, *, coords_bohr=None):
        x=jnp.asarray(outputs,dtype=self.parameters.centers.dtype)
        if x.shape != (len(self.symbols),2,2):
            raise ValueError('Contraction outputs must have shape (natom,2,2).')
        coords=self.parameters.nuclear_coords if coords_bohr is None else jnp.asarray(coords_bohr,dtype=x.dtype)
        if coords.shape != (len(self.symbols),3):
            raise ValueError('coords_bohr must have shape (natom,3).')
        cs=[]
        for atom,slot,c,b in zip(self.shell_atoms,self.slots,self.parameters.coefficients,self.tangents):
            if slot<0:
                cs.append(c)
            else:
                radius=jnp.linalg.norm(c[:,0])
                v=c[:,0]+radius*self.update_scale*(b@jnp.tanh(x[atom,slot]))
                cs.append((v*radius/jnp.linalg.norm(v))[:,None])
        return replace(self.parameters,coefficients=tuple(cs),nuclear_coords=coords,
                       centers=coords[jnp.asarray(self.shell_atoms)])

    def atom_shells(self, parameters):
        """Host-only export, one raw shell list per atom (never merges equal elements)."""
        records=[[] for _ in self.symbols]
        for atom,l,a,c in zip(self.shell_atoms,self.topology.angular_momenta,parameters.exponents,parameters.coefficients):
            rows=np.column_stack((np.asarray(a),np.asarray(c))).tolist()
            records[atom].append([l,*rows])
        return records


def prepare_basis(atom, *, unit='Angstrom',charge=0,spin=0,cart=True,update_scale=1.):
    """Create a static main-group SZP layout; all dynamic coordinates use Bohr."""
    if not np.isfinite(update_scale) or update_scale<=0:
        raise ValueError('update_scale must be finite and positive.')
    spec=atom if isinstance(atom,MoleculeSpec) else parse_molecule_spec(atom,unit=unit,charge=charge,spin=spin)
    if not np.array_equal(np.asarray(spec.charges),[atomic_number(s) for s in spec.symbols]):
        raise ValueError('NNAO templates require all-electron nuclear charges matching the elements.')
    ls=[]; atoms=[]; roles=[]; slots=[]; aa=[]; cc=[]; bb=[]
    for i,symbol in enumerate(spec.symbols):
        if symbol not in _templates():
            raise ValueError(f'NNAO SZP3 has no template for {symbol}; supported elements: {supported_elements()}')
        # Canonical AO order: angular momentum, then stable radial shell order.
        for shell in sorted(_templates()[symbol]['shells'],key=lambda item:item['l']):
            l=shell['l']; role=shell['role']
            a=jnp.asarray(shell['exponents'],dtype=jnp.float64)
            c=np.asarray(shell['coefficients'],dtype=float)
            slot={'valence_s':0,'valence_p':1}.get(role,-1)
            tangent=np.empty((len(c),0))
            if slot>=0:
                # A fixed orientation makes learned coordinates portable; an
                # SVD's degenerate nullspace may rotate between LAPACK builds.
                normal=c/np.linalg.norm(c)
                axis=np.eye(3)[np.argmin(np.abs(normal))]
                first=axis-normal*np.dot(axis,normal)
                first/=np.linalg.norm(first)
                tangent=np.column_stack((first,np.cross(normal,first)))
            ls.append(l); atoms.append(i);roles.append(role); slots.append(slot)
            aa.append(a);cc.append(jnp.asarray(c[:,None]));bb.append(jnp.asarray(tangent))
    coords=jnp.asarray(spec.coords_bohr,dtype=jnp.float64)
    top=BasisTopology(tuple(ls),tuple(len(a) for a in aa),(1,)*len(ls),tuple(int(z) for z in spec.charges),bool(cart))
    params=BasisParameters(tuple(aa),tuple(cc),coords[jnp.asarray(atoms)],coords)
    return NeuralBasis(top,params,tuple(spec.symbols),tuple(atoms),tuple(roles),tuple(slots),tuple(bb),float(update_scale))


@dataclass(frozen=True)
class DirectBasis:
    topology: BasisTopology
    parameters: BasisParameters
    symbols: tuple
    shell_atoms: tuple
    roles: tuple
    slots: tuple
    ecps: tuple
    charge: int=0
    spin: int=0
    max_primitives: int=5
    num_channels: int=3

    @property
    def nelectron(self):return sum(self.topology.nuclear_charges)-self.charge

    @property
    def exponent_scale_keys(self):
        """Element/angular-momentum keys for the nonfixed contractions present."""
        keys={(self.symbols[atom],l) for atom,l,slot in zip(
            self.shell_atoms,self.topology.angular_momenta,self.slots) if slot>=0}
        return tuple(sorted(keys,key=lambda item:(atomic_number(item[0]),item[1])))

    def with_log_exponent_scales(self,scales):
        """Return a new layout with alpha = current_alpha * exp(element/l scale).

        Core and other fixed shells are unchanged. Missing keys mean zero;
        invalid traced values produce NaNs instead of silently using a basis.
        """
        if not isinstance(scales,Mapping):raise ValueError('Exponent scales must be a mapping of (element,l) keys.')
        unknown=set(scales)-set(self.exponent_scale_keys)
        if unknown:raise ValueError(f'Unknown exponent scale keys for this layout: {unknown}')
        values={}
        for key,value in scales.items():
            value=jnp.asarray(value)
            if value.ndim!=0 or value.dtype.kind not in 'biuf':
                raise ValueError('Log exponent scales must be real scalars.')
            if not isinstance(value,jax.core.Tracer) and not np.isfinite(np.asarray(value)):
                raise ValueError('Log exponent scales must be finite.')
            values[key]=value
        exponents=[]
        for atom,l,slot,a in zip(self.shell_atoms,self.topology.angular_momenta,
                                self.slots,self.parameters.exponents):
            if slot<0:
                exponents.append(a)
                continue
            scale=jnp.asarray(values.get((self.symbols[atom],l),0.),dtype=a.dtype)
            scaled=a*jnp.exp(scale)
            valid=jnp.isfinite(scale)&jnp.all(jnp.isfinite(scaled))&jnp.all(scaled>0.)
            if not isinstance(valid,jax.core.Tracer) and not bool(valid):
                raise ValueError('Scaled Gaussian exponents must be finite and positive.')
            exponents.append(jnp.where(valid,scaled,jnp.nan))
        return replace(self,parameters=replace(self.parameters,exponents=tuple(exponents)))

    def reference_outputs(self):
        out=jnp.zeros((len(self.symbols),self.num_channels,self.max_primitives),dtype=self.parameters.centers.dtype)
        for atom,l,c in zip(self.shell_atoms,self.slots,self.parameters.coefficients):
            if l>=0:out=out.at[atom,l,:len(c)].set(c[:,0])
        return out

    def bind(self,outputs,*,coords_bohr=None):
        raw=jnp.asarray(outputs,dtype=self.parameters.centers.dtype)
        if raw.shape!=(len(self.symbols),self.num_channels,self.max_primitives):raise ValueError(f'Direct coefficients require shape {(len(self.symbols),self.num_channels,self.max_primitives)}.')
        cs=[]
        for atom,l,np_,fixed in zip(self.shell_atoms,self.slots,self.topology.primitive_counts,self.parameters.coefficients):
            if l<0:
                cs.append(fixed)
                continue
            c=raw[atom,l,:np_];norm=jnp.linalg.norm(c)
            if not isinstance(norm,jax.core.Tracer) and (not np.isfinite(norm) or norm<1e-12):
                raise ValueError('A direct contraction vector is zero or nonfinite.')
            cs.append((c/jnp.where(norm>=1e-12,norm,jnp.nan))[:,None])
        coords=self.parameters.nuclear_coords if coords_bohr is None else jnp.asarray(coords_bohr,dtype=raw.dtype)
        if coords.shape!=(len(self.symbols),3):raise ValueError('Coordinates must have shape (natom,3).')
        return replace(self.parameters,coefficients=tuple(cs),nuclear_coords=coords,centers=coords[jnp.asarray(self.shell_atoms)])

    atom_shells=NeuralBasis.atom_shells







def prepare_direct_basis(atom,*,unit='Angstrom',charge=0,spin=0,cart=False,
                         basis_family='szp442_direct',core_primitives=None):
    """All-electron direct contractions with fixed inner shells.

    szp442_direct expands p-block valence/polarization to 4s4p2d; the added
    exponent is half the previous minimum and its initial coefficient is zero.
    H/He and s-block templates retain their existing 3s + fixed 1p structure.
    szp3_direct explicitly selects the preceding 3s3p + fixed 1d comparison.
    szp663_direct expands p-block to 6s6p3d and H/He/s-block to 6s2p,
    retaining all old primitives and making the two-primitive p trainable.
    core_primitives defaults to 6 for szp442/663_direct, 3 for szp3_direct.
    Six-primitive cores use bundled 6-31G for Li-Ca; heavier elements require
    explicit core_primitives=3 until validated six-primitive data are supplied.
    """
    if core_primitives is None:core_primitives=3 if basis_family=='szp3_direct' else 6
    if core_primitives not in (3,6):raise ValueError('core_primitives must be 3 or 6.')
    spec=parse_molecule_spec(atom,unit=unit,charge=charge,spin=spin)
    if not np.array_equal(np.asarray(spec.charges),[atomic_number(s) for s in spec.symbols]):
        raise ValueError('Direct templates require all-electron physical nuclear charges.')
    data=_direct_templates(basis_family)
    ls=[];owners=[];roles=[];slots=[];aa=[];cc=[]
    for i,symbol in enumerate(spec.symbols):
        if symbol not in data:raise ValueError(f'NNAO has no direct template for {symbol}.')
        core_by_l={};core_index={}
        if core_primitives==6 and any(s['role']=='core' for s in data[symbol]['shells']):
            from gradscf.integrals.basis.data import load_basis_from_snapshot
            if atomic_number(symbol)>20:
                raise ValueError(f'No validated six-primitive core for {symbol}; select core_primitives=3 explicitly.')
            for block in load_basis_from_snapshot('6-31g',symbol):
                core_by_l.setdefault(block[0],[]).append(np.asarray(block[1:],dtype=float))
        for shell in sorted(data[symbol]['shells'],key=lambda x:x['l']):
            a=jnp.asarray(shell['exponents'],dtype=jnp.float64)
            c=jnp.asarray(shell['coefficients'],dtype=jnp.float64)[:,None]
            if shell['role']=='core' and core_primitives==6:
                l=shell['l'];index=core_index.get(l,0);core_index[l]=index+1
                rows=core_by_l[l][index]
                if rows.shape!=(6,2):raise ValueError(f'Invalid six-primitive core for {symbol}, l={l}, index={index}.')
                a=jnp.asarray(rows[:,0]);c=jnp.asarray(rows[:,1:])
            slot=_direct_slot(shell)
            ls.append(shell['l']);owners.append(i);roles.append(shell['role']);slots.append(slot)
            aa.append(a);cc.append(c/jnp.linalg.norm(c) if slot>=0 else c)
    coords=jnp.asarray(spec.coords_bohr,dtype=jnp.float64)
    topology=BasisTopology(tuple(ls),tuple(len(a) for a in aa),(1,)*len(ls),tuple(int(z) for z in spec.charges),bool(cart))
    parameters=BasisParameters(tuple(aa),tuple(cc),coords[jnp.asarray(owners)],coords)
    channels,primitives={'szp3_direct':(2,3),'szp442_direct':(3,4),'szp663_direct':(3,6)}[basis_family]
    return DirectBasis(topology,parameters,tuple(spec.symbols),tuple(owners),tuple(roles),tuple(slots),
                       (None,)*len(spec.symbols),spec.charge,spec.spin,primitives,channels)
