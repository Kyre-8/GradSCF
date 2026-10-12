# Equations and implementation sources

The general residual/Taylor route and its Wigner rank/energy completeness
argument are documented separately in [SERIES.md](SERIES.md). The MP2/MP3
equations below describe the independently retained specialized kernels.

The MP partition is defined by the physical HF Fock operator. Using HF normal
ordering, write

$$
H=E_{\mathrm{HF}}+F_N+\lambda V_N,
\qquad T_2=\lambda T_2^{(1)}+\lambda^2T_2^{(2)}+\cdots.
$$

For canonical RHF, first-order singles vanish by the Brillouin condition.
With spatial chemists' integrals and negative excitation denominators,

$$
\Delta_{ij}^{ab}=\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b,
\quad t_{ij}^{ab(1)}=\frac{(ia|jb)}{\Delta_{ij}^{ab}},
$$

$$
E^{(2)}=\sum_{ijab}t_{ij}^{ab(1)}[2(ia|jb)-(ib|ja)].
$$

The opposite-spin contribution is \(\sum t^{(1)}(ia|jb)\); the same-spin
contribution is \(\sum t^{(1)}[(ia|jb)-(ib|ja)]\). For UHF, same-spin amplitudes
use antisymmetrized integrals and energy factor 1/4 per spin block; the alpha-beta
block uses the direct integral and factor one. These spin factors count the
ordered tensor indices, not separate physical excitations multiple times.

For MP3 define the connected doubles action

$$
\mathcal A_V(X)_{ij}^{ab}
=\left.\frac{d}{d\eta}
\langle\Phi_{ij}^{ab}|e^{-\eta X}V_Ne^{\eta X}|\Phi_0\rangle
\right|_{\eta=0}.
$$

Expanding the connected doubles residual order by order gives

$$
0=V_{ij}^{ab}-\Delta_{ij}^{ab}t_{ij}^{ab(1)},
\qquad
0=\mathcal A_V(T_2^{(1)})_{ij}^{ab}-\Delta_{ij}^{ab}t_{ij}^{ab(2)}.
$$

Thus

$$
t_{ij}^{ab(2)}=\frac{\mathcal A_V(T_2^{(1)})_{ij}^{ab}}{\Delta_{ij}^{ab}},
\qquad
E^{(3)}=\sum_{ijab}t_{ij}^{ab(2)}[2(ia|jb)-(ib|ja)].
$$

`mp3.py` evaluates this action with `jax.jvp` of the existing restricted CCD
residual at zero amplitudes, setting its Fock part to zero. It does not truncate
a converged CC solution. Second-order singles do not enter E3 because Fov=0;
quadratic doubles terms first affect the residual at third order and the energy
at fourth order. The determinant-space test computes E2 and E3 independently
from the full Hamiltonian and diagonal Fock excitation energies.

Sources:

1. C. Moller and M. S. Plesset, *Note on an Approximation Treatment for
   Many-Electron Systems*, Physical Review **46**, 618-622 (1934),
   [DOI: 10.1103/PhysRev.46.618](https://doi.org/10.1103/PhysRev.46.618).
2. [PySCF restricted MP2 source](https://pyscf.org/_modules/pyscf/mp/mp2.html)
   and [unrestricted MP2 source](https://pyscf.org/_modules/pyscf/mp/ump2.html):
   forward validation baseline for energies, spin components and amplitudes.
   The GradSCF MP2 kernel is independently expressed as JAX contractions.
3. [GradSCF CC references](../cc/REFERENCES.md) and
   [CC adaptation notice](../cc/NOTICE.md): the licensed restricted CC residual
   reused by MP3 is already maintained in `gradscf.cc`; no duplicate residual
   or numerical solver is introduced in `gradscf.mp`.
4. [DePrince Lab perturbation tutorial](https://www.chem.fsu.edu/~deprince/tutorials/jupyter_notebooks/perturbation/perturbation.html):
   independent order-by-order Rayleigh--Schrodinger construction and intermediate
   normalization, underlying the determinant-space validation.
