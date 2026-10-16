"""Low-dimensional Coulomb kernels in atomic units.

These array kernels are building blocks, not a switch enabling a low-D SCF or
GW calculation. Ionic, electron-ion, exchange and q-cell head terms must use
the same boundary and potential gauge before a driver can expose them.

The truncations follow Ismail-Beigi, Phys. Rev. B 73, 233103 (2006),
https://doi.org/10.1103/PhysRevB.73.233103. The slab uses Eq. (3); the wire
integrates Eq. (5) over a rectangular transverse Wigner-Seitz cell. The wire
integral is evaluated with a Gaussian split to avoid a singular Bessel
quadrature. No upstream implementation is copied.
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc, exp1, log_ndtr


@jax.custom_jvp
def _positive_exp1(values):
    """Scalar branch evaluation with the analytic, higher-order E1 response."""
    return jax.lax.map(exp1, values)


@_positive_exp1.defjvp
def _positive_exp1_jvp(primals, tangents):
    values, = primals
    direction, = tangents
    return _positive_exp1(values), -jnp.exp(-values)/values*direction


def slab_kernel(vectors, lattice):
    """Fourier transform of theta(Lz/2-|z|)/r, shape (nG,).

    The last lattice row is normal to the first two. Wavevectors may include
    arbitrary in-plane q, but their normal component must be a reciprocal
    lattice integer. Rotation and strain within this geometry are supported.
    G=0 uses the finite-part gauge -pi*Lz**2/2, as in PySCF's slab kernel.
    This is not the divergent q->0 head average; q derivatives at Gamma are
    undefined. Invalid tilted cells/nonperiodic q return NaNs under JIT.
    """
    vectors, lattice = jnp.asarray(vectors), jnp.asarray(lattice)
    height = jnp.linalg.norm(lattice[2])
    normal = lattice[2]/height
    gz = vectors@normal
    planar = vectors-gz[:, None]*normal
    planar2 = jnp.sum(planar**2, axis=-1)
    gp = jnp.where(planar2 > 1e-24, jnp.sqrt(jnp.where(planar2 > 1e-24, planar2, 1.)), 0.)
    g2 = jnp.sum(vectors**2, axis=-1)
    integer = gz*height/(2*jnp.pi)
    parity = jnp.cos(jnp.pi*jnp.rint(integer))
    # expm1 preserves the small in-plane head away from Gamma.
    weight = (1-parity)-parity*jnp.expm1(-gp*height/2)
    kernel = jnp.where(g2 > 1e-24, 4*jnp.pi*weight/jnp.where(g2 > 1e-24, g2, 1.),
                       -jnp.pi*height**2/2)
    orthogonal = jnp.max(jnp.abs(lattice[:2]@normal)) < 1e-10
    valid = orthogonal & (jnp.abs(integer-jnp.rint(integer)) < 1e-9)
    return jnp.where(valid, kernel, jnp.nan)


def _real_energy(lattice, coordinates, charges, images, eta):
    dimension = images.shape[1]
    fractional = coordinates@jnp.linalg.inv(lattice)
    positions = coordinates-jnp.floor(fractional[:, :dimension])@lattice[:dimension]
    delta = positions[:, None]-positions[None, :]
    pair = charges[:, None]*charges[None, :]
    r2 = jnp.sum((delta[:, :, None]+(images@lattice[:dimension])[None, None])**2, axis=-1)
    radius = jnp.sqrt(jnp.where(r2 > 1e-24, r2, 1.))
    real = .5*jnp.sum(pair[:, :, None]*jnp.where(r2 > 1e-24, erfc(eta*radius)/radius, 0.))
    return delta, pair, real


def slab_energy(lattice, coordinates, charges, *, precision=1e-9, reference_lattice=None):
    """2D Ewald ion-ion energy, periodic in the first two lattice rows.

    The planar G=0 Green function is -2*pi*|z|/area. Its split Gaussian
    expression fixes the charged ionic subsystem's potential convention;
    electrons must use the matching slab kernel and local pseudopotential.
    Positions are wrapped only in-plane. Static image cutoffs come from the
    reference lattice, as in the existing 3D Ewald implementation.

    The reciprocal pair function is pi/G times
    exp(G*z)*erfc(G/(2*eta)+eta*z) + (z -> -z). Computing its exponential
    through log_ndtr avoids overflow before multiplication by erfc.
    """
    if not 0 < precision < 1:
        raise ValueError('precision must be in (0,1).')
    ref = np.asarray(lattice if reference_lattice is None else reference_lattice)
    ref_area = np.linalg.norm(np.cross(ref[0], ref[1]))
    eta = 2/np.sqrt(ref_area)
    cutoff = np.sqrt(-np.log(precision*1e-2))
    nr = int(np.ceil((cutoff/eta+np.linalg.norm(ref[:2], axis=1).sum())
                     /np.linalg.svd(ref[:2], compute_uv=False).min()))
    ref_reciprocal = 2*np.pi*np.linalg.inv(ref).T
    ng = int(np.ceil(2*eta*cutoff/np.linalg.svd(ref_reciprocal[:2], compute_uv=False).min()))
    images = jnp.asarray(list(itertools.product(range(-nr, nr+1), repeat=2)), dtype=float)
    integers = jnp.asarray([n for n in itertools.product(range(-ng, ng+1), repeat=2)
                            if n != (0, 0)], dtype=float)
    lattice, coordinates, charges = map(jnp.asarray, (lattice, coordinates, charges))
    normal = lattice[2]/jnp.linalg.norm(lattice[2])
    area = jnp.linalg.norm(jnp.cross(lattice[0], lattice[1]))
    delta, pair, real = _real_energy(lattice, coordinates, charges, images, eta)
    gv = integers@(2*jnp.pi*jnp.linalg.inv(lattice).T)[:2]
    g = jnp.linalg.norm(gv, axis=-1)
    z = delta@normal
    def damped(sign):
        argument = g/(2*eta)+sign*eta*z[:, :, None]
        return jnp.exp(sign*z[:, :, None]*g+log_ndtr(-jnp.sqrt(2.)*argument)+jnp.log(2.))
    reciprocal = jnp.sum(pair[:, :, None]*jnp.cos(delta@gv.T)*jnp.pi/g*(damped(1)+damped(-1)))
    zero = -2*jnp.pi*(z*erf(eta*z)+jnp.exp(-(eta*z)**2)/(eta*jnp.sqrt(jnp.pi)))
    result = real-eta/jnp.sqrt(jnp.pi)*jnp.sum(charges**2)+(reciprocal+jnp.sum(pair*zero))/(2*area)
    return jnp.where(jnp.max(abs(lattice[:2]@normal)) < 1e-10, result, jnp.nan)


def wire_energy(lattice, coordinates, charges, *, precision=1e-9,
                quadrature=96, reference_lattice=None):
    """One-periodic-direction Ewald energy, in the wire_kernel finite-part gauge.

    Longitudinal G=0 uses the Green function -2*log(rho/bohr). Subtracting
    the t=0 Gaussian before quadrature isolates its logarithmic finite part.
    The result is independent of the two vacuum lengths. There is no
    artificial transverse ionic image sum. Image topology is static during
    geometry AD and the real-space sum is shared with slab_energy.
    """
    if not 0 < precision < 1 or not isinstance(quadrature, int) or quadrature < 16:
        raise ValueError('Use precision in (0,1) and quadrature >=16.')
    ref = np.asarray(lattice if reference_lattice is None else reference_lattice)
    ref_length = np.linalg.norm(ref[0])
    eta = 2/ref_length
    cutoff = np.sqrt(-np.log(precision*1e-2))
    nr = int(np.ceil(cutoff/(eta*ref_length)))+2
    ng = int(np.ceil(eta*cutoff*ref_length/np.pi))
    images = jnp.arange(-nr, nr+1, dtype=float)[:, None]
    integers = jnp.arange(-ng, ng+1, dtype=float)
    lattice, coordinates, charges = map(jnp.asarray, (lattice, coordinates, charges))
    length = jnp.linalg.norm(lattice[0])
    axis = lattice[0]/length
    delta, pair, real = _real_energy(lattice, coordinates, charges, images, eta)
    x = delta@axis
    transverse = delta-x[:, :, None]*axis
    rho2 = jnp.sum(transverse**2, axis=-1)
    g = integers*2*jnp.pi/length
    nodes, weights = map(jnp.asarray, np.polynomial.legendre.leggauss(quadrature))
    t, tw = eta*(nodes+1)/2, eta*weights/2
    remainder = jnp.expm1(-rho2[:, :, None]*t**2)
    exponential = jnp.exp(-g[:, None]**2/(4*t**2))
    long = jnp.einsum('t,gt,ijt->ijg', 2*tw/t, exponential, remainder)
    argument = jnp.where(integers != 0, g**2/(4*eta**2), 1.)
    head = jnp.where(integers != 0, _positive_exp1(argument), jnp.log(eta**2)+jnp.euler_gamma)
    long += head
    reciprocal = jnp.sum(pair[:, :, None]*jnp.cos(x[:, :, None]*g)*long)/(2*length)
    result = real-eta/jnp.sqrt(jnp.pi)*jnp.sum(charges**2)+reciprocal
    lengths = jnp.linalg.norm(lattice, axis=1)
    gram = lattice@lattice.T/(lengths[:, None]*lengths[None, :])
    return jnp.where(jnp.max(abs(gram-jnp.eye(3))) < 1e-10, result, jnp.nan)


def wire_kernel(lattice, mesh, *, wave=0., quadrature=96, precision=1e-12):
    """Rectangular Wigner-Seitz wire kernel on a 3D FFT grid.

    The first lattice row is the periodic axis; all rows are orthogonal.
    ``wave`` is its signed longitudinal momentum shift in inverse bohr.
    ``mesh`` contains odd sizes; the result has exactly that shape and NumPy
    FFT ordering. Lattice/wave remain JAX operands; quadrature is static.

    Splitting 1/r into erfc(eta*r)/r + erf(eta*r)/r gives the short-range
    transform 4*pi/G²*(1-exp(-G²/(4*eta²))). The long-range transform is
    2*integral_0^eta exp(-kx²/(4*t²))*Iy(t)*Iz(t)/t dt. Each transverse
    Gaussian Fourier integral is separated into its exact t=0 value and an
    expm1 remainder. Only the transverse head diverges as kx->0; its analytic
    integral is area*E1(kx²/(4*eta²)). At kx=0 the chosen finite-part gauge is
    the Fourier transform of -2*log(rho/bohr). It is not a q-cell average.

    The omitted short-range tail outside the rectangle is controlled by
    ``precision`` and can be checked by tightening it and the quadrature.
    This does not control subsequent SCF/GW/basis/FFT errors. Gamma momentum
    differentiation is not defined; geometry derivatives at fixed Gamma are.
    """
    if len(mesh) != 3 or any(int(n) != n or n < 3 or n % 2 != 1 for n in mesh):
        raise ValueError('Use three odd FFT mesh sizes >=3.')
    if not isinstance(quadrature, int) or quadrature < 16 or not 0 < precision < 1:
        raise ValueError('Use quadrature >=16 and precision in (0,1).')
    lattice = jnp.asarray(lattice)
    lengths = jnp.linalg.norm(lattice, axis=1)
    lx, ly, lz = lengths
    indices = [jnp.asarray(np.fft.fftfreq(n)*n) for n in mesh]
    kx = indices[0]*2*jnp.pi/lx+wave
    gy, gz = indices[1]*2*jnp.pi/ly, indices[2]*2*jnp.pi/lz
    eta = jnp.sqrt(-jnp.log(precision))/(jnp.minimum(ly, lz)/2)
    nodes, weights = map(jnp.asarray, np.polynomial.legendre.leggauss(quadrature))
    t, tw = eta*(nodes+1)/2, eta*weights/2

    def remainder(length, integers):
        position = length*(nodes+1)/4
        phase = jnp.pi/2*(nodes[:, None]+1)*integers[None, :]
        return length/2*jnp.einsum('u,tu,un->tn', weights,
            jnp.expm1(-(t[:, None]*position[None, :])**2), jnp.cos(phase))

    dy, dz = remainder(ly, indices[1]), remainder(lz, indices[2])
    hy, hz = (indices[1] == 0).astype(lattice.dtype), (indices[2] == 0).astype(lattice.dtype)
    regular = dy[:, :, None]*dz[:, None, :]+ly*hy[None, :, None]*dz[:, None, :]
    regular += lz*hz[None, None, :]*dy[:, :, None]
    longitudinal = jnp.exp(-kx[None, :]**2/(4*t[:, None]**2))
    long = jnp.einsum('t,tx,tyz->xyz', 2*tw/t, longitudinal, regular)
    nonzero = kx**2 > 1e-24
    argument = jnp.where(nonzero, kx**2/(4*eta**2), 1.)
    # Scalar control flow avoids evaluating the continued-fraction E1 branch
    # at tiny x under the library's vectorized piecewise implementation.
    head = jnp.where(nonzero, _positive_exp1(argument), jnp.log(eta**2)+jnp.euler_gamma)
    long += ly*lz*head[:, None, None]*hy[None, :, None]*hz[None, None, :]
    g2 = kx[:, None, None]**2+gy[None, :, None]**2+gz[None, None, :]**2
    safe = jnp.where(g2 > 1e-24, g2, 1.)
    short = jnp.where(g2 > 1e-24, -4*jnp.pi*jnp.expm1(-g2/(4*eta**2))/safe, jnp.pi/eta**2)
    gram = lattice@lattice.T/(lengths[:, None]*lengths[None, :])
    return jnp.where(jnp.max(abs(gram-jnp.eye(3))) < 1e-10, short+long, jnp.nan)
