"""Analytic Cartesian Gaussian Fourier coefficients for Bloch AO grids."""
import numpy as np
import jax
import jax.numpy as jnp
from ..basis import cartesian_angular_tuples
from gradscf.integrals.basis.normalization import normalized_shell_coefficients, radial_primitive_norm
from gradscf.integrals.basis.normalization import primitive_cartesian_norm


def reciprocal_grid(lattice,mesh):
    indices=np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in mesh],indexing='ij'),axis=-1).reshape(-1,3)
    return jnp.asarray(indices)@(2*jnp.pi*jnp.linalg.inv(lattice).T)


def fourier_ao(topology,parameters,vectors):
    columns=[]
    for shell,l in enumerate(topology.angular_momenta):
        a=parameters.exponents[shell]
        coeff=normalized_shell_coefficients(l,a,parameters.coefficients[shell])/radial_primitive_norm(l,a)[:,None]
        q=vectors[:,None,:]
        gaussian=(jnp.pi/a)[None,:]**1.5*jnp.exp(-jnp.sum(q*q,axis=-1)/(4*a))
        phase=jnp.exp(-1j*vectors@parameters.centers[shell])
        for contraction in range(coeff.shape[1]):
            for angular in cartesian_angular_tuples(l):
                moment=jnp.ones_like(gaussian,dtype=complex)
                for axis,power in enumerate(angular):
                    prev=jnp.ones_like(gaussian,dtype=complex)
                    cur=-1j*q[:,:,axis]/(2*a)
                    for order in range(1,power):
                        prev,cur=cur,-1j*q[:,:,axis]/(2*a)*cur+order/(2*a)*prev
                    moment=moment*(prev if power==0 else cur)
                columns.append(phase*jnp.sum(gaussian*moment*coeff[:,contraction]*primitive_cartesian_norm(a,angular),axis=1))
    return jnp.stack(columns,axis=1)


def bloch_grid(topology,parameters,lattice,mesh,kpoints):
    """Return (nk,4,ngrid,nao): value and Cartesian first derivatives."""
    gv=reciprocal_grid(lattice,mesh);volume=jnp.linalg.det(lattice);ngrid=np.prod(mesh)
    frac=np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh],indexing='ij'),axis=-1).reshape(-1,3)
    coords=jnp.asarray(frac)@lattice
    def at_k(k):
        q=gv+k;ft=fourier_ao(topology,parameters,q)
        phase=jnp.exp(1j*coords@k)
        channels=[ft]+[1j*q[:,axis,None]*ft for axis in range(3)]
        grid=jnp.stack([jnp.fft.ifftn((values*ngrid/volume).reshape(tuple(mesh)+(-1,)),axes=(0,1,2)).reshape(values.shape)*phase[:,None] for values in channels])
        return grid,ft
    grids,fourier=jax.vmap(at_k)(kpoints)
    return grids,fourier,gv,coords
