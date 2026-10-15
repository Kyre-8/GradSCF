"""Scalar Cartesian space-group maps in the lattice-translation Bloch gauge.

Lattice vectors are rows. Fractional positions transform as x' = W x + t
(column convention), and fractional reciprocal rows as k' = k W^-1.
Discovery/mapping is eager; coefficient, density and operator maps support JAX.
These are physical spatial operations, not an irreducible-SCF implementation.
"""
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
from ..integrals.basis import cartesian_angular_tuples


def _readonly(value,dtype=None):
    value=np.array(value,dtype=dtype,copy=True)
    value.setflags(write=False)
    return value


def _freeze(value):
    if isinstance(value,(list,tuple,np.ndarray)):
        return tuple(_freeze(v) for v in value)
    return value


def _lattice(cell):
    a=np.asarray(cell.lattice)
    if a.shape!=(3,3) or np.iscomplexobj(a) or not np.isfinite(a).all():
        raise ValueError('Lattice must be a finite real 3x3 matrix.')
    determinant=np.linalg.det(a)
    if not np.isfinite(determinant) or determinant<=0:
        raise ValueError('Lattice must be finite, nonsingular and right-handed.')
    return a


def _atom_data(cell,tolerance):
    if not cell.cart or not cell.topology.cart or cell.dimension!=3:
        raise NotImplementedError('Space-group AO maps require scalar 3D Cartesian cells.')
    from ..integrals.basis.types import check_parameter_shapes
    check_parameter_shapes(cell.topology,cell.parameters)
    coords=np.asarray(cell.coords);p=cell.parameters
    if (coords.shape!=p.nuclear_coords.shape or any(np.iscomplexobj(x) or not np.isfinite(x).all()
            for x in (coords,p.centers,p.nuclear_coords))):
        raise ValueError('Atomic and basis coordinates must be finite real (n,3) arrays.')
    if not np.allclose(p.nuclear_coords,coords,atol=tolerance,rtol=0):
        raise ValueError('Cell and basis nuclear coordinates disagree.')
    distances=np.linalg.norm(np.asarray(p.centers)[:,None]-coords[None],axis=-1)
    owners=distances.argmin(axis=1)
    if np.any(distances[np.arange(len(owners)),owners]>tolerance):
        raise ValueError('Symmetry requires atom-attached basis centers.')
    shell_keys=[]
    for l,a,c in zip(cell.topology.angular_momenta,p.exponents,p.coefficients):
        a,c=np.asarray(a),np.asarray(c)
        if (np.iscomplexobj(a) or np.iscomplexobj(c) or not np.isfinite(a).all()
                or np.any(a<=0) or not np.isfinite(c).all()):
            raise ValueError('Basis parameters must be finite and real with positive exponents.')
        shell_keys.append((l,len(a),c.shape[1],tuple(a.tolist()),tuple(c.ravel().tolist())))
    shells=[np.flatnonzero(owners==i).tolist() for i in range(len(coords))]
    keys=[(cell.topology.nuclear_charges[i],_freeze(cell.pseudopotentials[i]),
           tuple(sorted(shell_keys[s] for s in ids))) for i,ids in enumerate(shells)]
    return owners,shells,shell_keys,keys


def _signature(cell):
    from ..scf.reference import _array_signature
    p=cell.parameters
    arrays=(cell.lattice,cell.coords,p.centers,p.nuclear_coords,*p.exponents,*p.coefficients)
    return (cell.topology,tuple(_array_signature(x) for x in arrays),
            _freeze(cell.pseudopotentials),cell.cart,cell.dimension)


def cartesian_rotation(l,rotation):
    """Polynomial representation m_l(r Q) = m_l(r) D_l(Q).

    Ordering and normalization match the native Cartesian AOs: all components
    of a shell share a radial factor. For l >= 2, D is generally not unitary
    in the Euclidean coefficient metric. Reflections/inversion are included.
    """
    q=np.asarray(rotation)
    if (isinstance(l,bool) or not isinstance(l,(int,np.integer)) or l<0
            or q.shape!=(3,3) or np.iscomplexobj(q) or not np.isfinite(q).all()
            or not np.allclose(q.T@q,np.eye(3),atol=1e-7,rtol=0)):
        raise ValueError('Use a nonnegative angular momentum and a real orthogonal 3x3 rotation.')
    angular=cartesian_angular_tuples(l);indices={a:i for i,a in enumerate(angular)}
    result=np.zeros((len(angular),len(angular)))
    for column,powers in enumerate(angular):
        polynomial={(0,0,0):1.}
        for axis,power in enumerate(powers):
            for _ in range(power):
                product={}
                for monomial,coefficient in polynomial.items():
                    for coordinate in range(3):
                        exponent=list(monomial);exponent[coordinate]+=1;exponent=tuple(exponent)
                        product[exponent]=product.get(exponent,0.)+coefficient*q[coordinate,axis]
                polynomial=product
        for monomial,coefficient in polynomial.items():
            result[indices[monomial],column]=coefficient
    return result


@dataclass(frozen=True,eq=False)
class Operation:
    """Validated immutable map for one cell; every k argument is fractional.

    If transformed atom i equals atom j plus integer L_i, the AO coefficient
    map has phase exp(-2 pi i k'.L_i). Coefficients and densities transform as
    C'=T C and P'=T P T^dagger, whereas operators use T^-dagger F T^-1.
    """
    rotation: object
    translation: object
    cartesian: object
    atom_map: object
    lattice_shifts: object
    _ao_rotation: object=field(repr=False)
    _ao_shifts: object=field(repr=False)
    _signature: object=field(repr=False)

    @classmethod
    def from_cell(cls,cell,rotation,translation=(0.,0.,0.),*,tolerance=1e-7):
        w=np.asarray(rotation);t=np.asarray(translation)
        if (np.ndim(tolerance)!=0 or not np.isfinite(tolerance) or tolerance<=0
                or w.shape!=(3,3) or t.shape!=(3,) or np.iscomplexobj(w) or np.iscomplexobj(t)
                or not np.isfinite(w).all() or not np.isfinite(t).all()):
            raise ValueError('Use a finite real rotation/translation and positive tolerance.')
        if not np.allclose(w,np.rint(w),atol=1e-10,rtol=0):
            raise ValueError('Fractional space-group rotations must be integral.')
        w=np.rint(w).astype(int)
        if abs(round(np.linalg.det(w)))!=1:
            raise ValueError('Fractional rotation must be unimodular.')
        a=_lattice(cell);q=a.T@w@np.linalg.inv(a.T)
        if not np.allclose(q.T@q,np.eye(3),atol=tolerance/max(1.,np.linalg.norm(a)),rtol=0):
            raise ValueError('Operation is not an isometry of this lattice.')
        _,shells,shell_keys,keys=_atom_data(cell,tolerance)
        fractional=np.asarray(cell.coords)@np.linalg.inv(a)
        moved=fractional@w.T+t
        mapping=[];shifts=[]
        for i,point in enumerate(moved):
            difference=point-fractional;integer=np.rint(difference)
            matches=np.flatnonzero(np.linalg.norm((difference-integer)@a,axis=1)<=tolerance)
            if len(matches)!=1:
                raise ValueError('Operation must map atoms uniquely modulo lattice translations.')
            j=int(matches[0])
            if keys[i]!=keys[j]:
                raise ValueError('Operation changes atomic species, basis or pseudopotential.')
            mapping.append(j);shifts.append(integer[j].astype(int))
        if len(set(mapping))!=len(mapping):raise ValueError('Atom map must be a permutation.')
        sizes=[(l+1)*(l+2)//2*nc for l,nc in zip(cell.topology.angular_momenta,cell.topology.contraction_counts)]
        offsets=np.cumsum([0]+sizes);nao=cell.topology.nao
        transform=np.zeros((nao,nao));ao_shifts=np.zeros((nao,3),dtype=int)
        rotations={l:cartesian_rotation(l,q) for l in set(cell.topology.angular_momenta)}
        for i,j in enumerate(mapping):
            available=list(shells[j])
            for source in shells[i]:
                target=next(s for s in available if shell_keys[s]==shell_keys[source])
                available.remove(target)
                block=rotations[cell.topology.angular_momenta[source]]
                count=cell.topology.contraction_counts[source]
                transform[offsets[target]:offsets[target+1],offsets[source]:offsets[source+1]]=np.kron(np.eye(count),block)
                ao_shifts[offsets[source]:offsets[source+1]]=shifts[i]
        return cls(*map(_readonly,(w,t,q,mapping,shifts,transform,ao_shifts)),_signature(cell))

    def validate(self,cell):
        """Reject reusing this static map after geometry, basis or potential edits."""
        if _signature(cell)!=self._signature:
            raise ValueError('Cell geometry, basis or pseudopotential changed; rebuild the symmetry map.')

    def rotate_k(self,k):
        """Rotate a fractional reciprocal vector without a discontinuous wrap."""
        k=jnp.asarray(k)
        if k.shape!=(3,) or jnp.iscomplexobj(k):raise ValueError('k must be a real fractional three-vector.')
        if not isinstance(k,jax.core.Tracer) and not np.isfinite(k).all():
            raise ValueError('k must be finite.')
        return k@jnp.asarray(np.rint(np.linalg.inv(self.rotation)).astype(int))

    def matrix(self,k):
        phase=jnp.exp(-2j*jnp.pi*(jnp.asarray(self._ao_shifts)@self.rotate_k(k)))
        return jnp.asarray(self._ao_rotation)*phase[None,:]

    def coefficients(self,k,coefficients):
        c=jnp.asarray(coefficients);t=self.matrix(k)
        if c.ndim<2 or c.shape[-2]!=len(t):raise ValueError('Coefficients require (...,nao,nmo).')
        return t@c

    def density(self,k,density):
        d=jnp.asarray(density);t=self.matrix(k)
        if d.ndim<2 or d.shape[-2:]!=t.shape:raise ValueError('Density requires (...,nao,nao).')
        return t@d@t.conj().T

    def operator(self,k,operator):
        f=jnp.asarray(operator);t=self.matrix(k)
        if f.ndim<2 or f.shape[-2:]!=t.shape:raise ValueError('Operator requires (...,nao,nao).')
        inverse=jnp.linalg.inv(t)
        return inverse.conj().T@f@inverse


def space_group(cell,*,tolerance=1e-7):
    """Discover spatial operations with optional spglib; tolerance is in Bohr.

    Actual radial basis and GTH data distinguish atomic types conservatively.
    Magnetic order, external fields and FFT-cutoff symmetry are not inferred.
    """
    if np.ndim(tolerance)!=0 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('tolerance must be a finite positive length in Bohr.')
    try:
        import spglib
    except ImportError as error:
        raise ImportError('Install gradscf[symmetry] for discovery, or supply explicit Operation maps.') from error
    a=_lattice(cell)
    _,_,_,keys=_atom_data(cell,tolerance)
    unique={};labels=[unique.setdefault(key,len(unique)+1) for key in keys]
    fractional=np.asarray(cell.coords)@np.linalg.inv(a)
    result=spglib.get_symmetry((a,fractional,labels),symprec=tolerance)
    if result is None:raise RuntimeError('spglib could not determine the spatial symmetry.')
    return tuple(Operation.from_cell(cell,w,t,tolerance=tolerance)
                 for w,t in zip(result['rotations'],result['translations']))
